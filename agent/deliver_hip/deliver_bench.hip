// Deliver strategies on a recorded spike trace (dump_spikes.py), replicated over C columns.
// A  atomic, Mojo's shiu_worker scheme: 32 lanes per spike stride its out-edges, a fixed
//    1024 x 256 grid strides the spikes, atomicAdd into g.
// B  flat atomic: exclusive scan of out-degrees, one thread per edge, atomicAdd into g.
// C  sort-reduce: expand (target, w) pairs, rocprim radix_sort_pairs by target,
//    reduce_by_key, one plain add per distinct target (no atomics, deterministic order).
// Refractory masking and edge groups are left out: the same per-edge cost in all three.
// Per-step edge totals for B and C launch sizes come from the host trace, outside the timing.
// usage: deliver_bench GRAPH_DIR SPIKE_DIR C      prints us/step per strategy and max |g - g_A|
#include <hip/hip_runtime.h>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/device/device_reduce_by_key.hpp>
#include <rocprim/device/device_scan.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "FAIL %s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

template <class T> std::vector<T> load(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "FAIL load %s\n", path.c_str()); exit(1); }
    std::vector<T> v(f.tellg() / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(T));
    return v;
}

__global__ void deliver_lanes(const int *row, const int *col, const float *w, const int *spk, int spikes, int n, float *g) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, lane = k % 32;
    for (int s = k / 32; s < spikes; s += gridDim.x * blockDim.x / 32) {
        int i = spk[s], base = (i / n) * n, pre = i - base;
        for (int e = row[pre] + lane; e < row[pre + 1]; e += 32) atomicAdd(&g[base + col[e]], w[e]);
    }
}

__global__ void degree(const int *row, const int *spk, int spikes, int n, int *deg) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < spikes) { int pre = spk[s] % n; deg[s] = row[pre + 1] - row[pre]; }
}

// the spike owning flat edge e: binary search over the scanned degrees
__device__ int owner(const int *start, int spikes, int e) {
    int lo = 0, hi = spikes - 1;
    while (lo < hi) { int mid = (lo + hi + 1) / 2; if (start[mid] <= e) lo = mid; else hi = mid - 1; }
    return lo;
}

__global__ void deliver_flat(const int *row, const int *col, const float *w, const int *spk, const int *start, int spikes, int edges, int n, float *g) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    int s = owner(start, spikes, e), i = spk[s], base = (i / n) * n, k = row[i - base] + e - start[s];
    atomicAdd(&g[base + col[k]], w[k]);
}

__global__ void expand(const int *row, const int *col, const float *w, const int *spk, const int *start, int spikes, int edges, int n, int *key, float *val) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    int s = owner(start, spikes, e), i = spk[s], base = (i / n) * n, k = row[i - base] + e - start[s];
    key[e] = base + col[k];
    val[e] = w[k];
}

__global__ void scatter(const int *key, const float *sum, const int *unique, float *g) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < *unique) g[key[u]] += sum[u];
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: deliver_bench GRAPH_DIR SPIKE_DIR C\n"); return 1; }
    std::string gdir = argv[1], sdir = argv[2];
    int C = atoi(argv[3]);
    auto row = load<int>(gdir + "/row_ptr.bin"); auto col = load<int>(gdir + "/col_idx.bin"); auto w = load<float>(gdir + "/weights.bin");
    auto spikes1 = load<int>(sdir + "/spikes.bin"); auto off1 = load<int>(sdir + "/offsets.bin");
    int n = row.size() - 1, steps = off1.size() - 1;
    long S = (long)n * C;
    std::vector<int> spk, off{0}, edges_at(steps);  // column c's copy of a spike is neuron + c * n
    int maxsp = 0, maxe = 0;
    for (int t = 0; t < steps; t++) {
        for (int c = 0; c < C; c++) for (int k = off1[t]; k < off1[t + 1]; k++) spk.push_back(spikes1[k] + c * n);
        off.push_back(spk.size());
        long e = 0;
        for (int k = off[t]; k < off[t + 1]; k++) e += row[spk[k] % n + 1] - row[spk[k] % n];
        edges_at[t] = (int)e;
        maxsp = std::max(maxsp, off[t + 1] - off[t]);
        maxe = std::max(maxe, (int)e);
    }
    int *d_row, *d_col, *d_spk, *d_deg, *d_start, *d_key, *d_key2, *d_ukey, *d_unique;
    float *d_w, *d_val, *d_val2, *d_sum, *g[3];
    CHECK(hipMalloc(&d_row, row.size() * 4)); CHECK(hipMalloc(&d_col, col.size() * 4)); CHECK(hipMalloc(&d_w, w.size() * 4));
    CHECK(hipMalloc(&d_spk, spk.size() * 4)); CHECK(hipMalloc(&d_deg, maxsp * 4 + 4)); CHECK(hipMalloc(&d_start, maxsp * 4 + 4));
    for (auto &x : g) CHECK(hipMalloc(&x, S * 4));
    CHECK(hipMalloc(&d_key, maxe * 4 + 4)); CHECK(hipMalloc(&d_key2, maxe * 4 + 4)); CHECK(hipMalloc(&d_ukey, maxe * 4 + 4));
    CHECK(hipMalloc(&d_val, maxe * 4 + 4)); CHECK(hipMalloc(&d_val2, maxe * 4 + 4)); CHECK(hipMalloc(&d_sum, maxe * 4 + 4));
    CHECK(hipMalloc(&d_unique, 4));
    CHECK(hipMemcpy(d_row, row.data(), row.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_col, col.data(), col.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_w, w.data(), w.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_spk, spk.data(), spk.size() * 4, hipMemcpyHostToDevice));
    size_t scan_b = 0, sort_b = 0, red_b = 0;  // rocprim temp storage sized for the largest step
    CHECK(rocprim::exclusive_scan(nullptr, scan_b, d_deg, d_start, 0, maxsp + 1, rocprim::plus<int>()));
    CHECK(rocprim::radix_sort_pairs(nullptr, sort_b, d_key, d_key2, d_val, d_val2, (unsigned)maxe, 0, 32));
    CHECK(rocprim::reduce_by_key(nullptr, red_b, d_key2, d_val2, (unsigned)maxe, d_ukey, d_sum, d_unique, rocprim::plus<float>(), rocprim::equal_to<int>()));
    size_t tmp_b = std::max(scan_b, std::max(sort_b, red_b));
    void *tmp;
    CHECK(hipMalloc(&tmp, tmp_b));
    int bits = 32 - __builtin_clz((unsigned)S);
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    float ms[3];
    for (int strat = 0; strat < 3; strat++) {
        CHECK(hipMemset(g[strat], 0, S * 4));
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(a));
        for (int t = 0; t < steps; t++) {
            int ns = off[t + 1] - off[t], edges = edges_at[t];
            if (ns == 0 || edges == 0) continue;
            const int *sp = d_spk + off[t];
            if (strat == 0) { hipLaunchKernelGGL(deliver_lanes, dim3(1024), dim3(256), 0, 0, d_row, d_col, d_w, sp, ns, n, g[0]); continue; }
            hipLaunchKernelGGL(degree, dim3((ns + 255) / 256), dim3(256), 0, 0, d_row, sp, ns, n, d_deg);
            size_t sb = tmp_b;
            CHECK(rocprim::exclusive_scan(tmp, sb, d_deg, d_start, 0, ns, rocprim::plus<int>()));
            if (strat == 1) { hipLaunchKernelGGL(deliver_flat, dim3((edges + 255) / 256), dim3(256), 0, 0, d_row, d_col, d_w, sp, d_start, ns, edges, n, g[1]); continue; }
            hipLaunchKernelGGL(expand, dim3((edges + 255) / 256), dim3(256), 0, 0, d_row, d_col, d_w, sp, d_start, ns, edges, n, d_key, d_val);
            sb = tmp_b;
            CHECK(rocprim::radix_sort_pairs(tmp, sb, d_key, d_key2, d_val, d_val2, (unsigned)edges, 0, bits));
            sb = tmp_b;
            CHECK(rocprim::reduce_by_key(tmp, sb, d_key2, d_val2, (unsigned)edges, d_ukey, d_sum, d_unique, rocprim::plus<float>(), rocprim::equal_to<int>()));
            hipLaunchKernelGGL(scatter, dim3((edges + 255) / 256), dim3(256), 0, 0, d_ukey, d_sum, d_unique, g[2]);
        }
        CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b)); CHECK(hipGetLastError());
        CHECK(hipEventElapsedTime(&ms[strat], a, b));
    }
    std::vector<float> ga(S), gx(S);
    CHECK(hipMemcpy(ga.data(), g[0], S * 4, hipMemcpyDeviceToHost));
    const char *name[3] = {"A atomic lanes (Mojo)", "B flat atomic", "C sort-reduce"};
    int bad = 0;
    for (int strat = 0; strat < 3; strat++) {
        double err = 0, mag = 0;
        if (strat) {
            CHECK(hipMemcpy(gx.data(), g[strat], S * 4, hipMemcpyDeviceToHost));
            for (long i = 0; i < S; i++) { err = std::max(err, (double)fabsf(gx[i] - ga[i])); mag = std::max(mag, (double)fabsf(ga[i])); }
        }
        bool ok = err <= 1e-5 * std::max(mag, 1.0);
        bad += !ok;
        printf("%-24s %8.1f us/step   max |g - g_A| %.3g (of %.3g) %s\n", name[strat], ms[strat] * 1000 / steps, err, mag, ok ? "PASS" : "FAIL");
    }
    printf("%d columns, %d steps, max %d spikes / %d edges per step\n", C, steps, maxsp, maxe);
    return bad ? 1 : 0;
}

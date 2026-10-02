#!/usr/bin/env mojo
"""Tiny CPU probe: can a learned readout separate two synthetic inputs
after activity propagates through the fixed FlyWire CSR graph?

This is an engineering probe, not a biologically mapped sensory experiment.
Run from this directory: mojo readout_probe.mojo
"""

from std.collections import List
from max.algorithm.backend.cpu.parallelize import sync_parallelize

comptime N = 139255
comptime STEPS = 4
comptime TRAIN = 32
comptime TEST = 64

def zeros(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(0.0)
    return out^

def run_sample(
    row_ptr: Pointer[Int32, _],
    col_idx: Pointer[Int32, _],
    weights: Pointer[Float32, _],
    condition: Int,
    sample_id: Int,
) -> List[Float32]:
    var x = zeros(N)
    var y = zeros(N)
    var v = zeros(N)
    var features = zeros(N)

    var x_ptr = x.unsafe_ptr()
    var y_ptr = y.unsafe_ptr()
    var v_ptr = v.unsafe_ptr()
    for step in range(STEPS):
        # A then B versus B then A. At readout time (step 3), both classes
        # receive no external drive. The index groups are synthetic, not known
        # biological sensory populations.
        var symbol = -1
        if step == 0:
            symbol = 0 if condition == 0 else 1
        elif step == 2:
            symbol = 1 if condition == 0 else 0
        if symbol >= 0:
            for k in range(96):
                var pattern_hash = k * 7919 + sample_id * 104729 + k * sample_id * 37
                if pattern_hash % 100 >= 20:
                    var idx = (k * 7919 + 17 + symbol * (N // 2)) % N
                    x[idx] = 1.0
        def lif_row(r: Int) {imm}:
            var acc: Float32 = 0.0
            var k = Int(row_ptr[unsafe_offset=r])
            var end = Int(row_ptr[unsafe_offset=r + 1])
            while k < end:
                var c = Int(col_idx[unsafe_offset=k])
                if x_ptr[unsafe_offset=c] > 0.0:
                    acc += weights[unsafe_offset=k] * x_ptr[unsafe_offset=c]
                k += 1
            var membrane = 0.85 * v_ptr[unsafe_offset=r] + 0.08 * acc
            if membrane >= 1.0:
                y_ptr[unsafe_offset=r] = 1.0
                v_ptr[unsafe_offset=r] = 0.0
            else:
                y_ptr[unsafe_offset=r] = 0.0
                if membrane < -5.0:
                    v_ptr[unsafe_offset=r] = -5.0
                else:
                    v_ptr[unsafe_offset=r] = membrane

        sync_parallelize(lif_row, N)
        for i in range(N):
            if step == STEPS - 1:
                features[i] = y[i] + v[i] * 0.1
            x[i] = y[i]
    return features^

def predict(feature: List[Float32], proto0: List[Float32], proto1: List[Float32]) -> Int:
    var d0: Float64 = 0.0
    var d1: Float64 = 0.0
    for i in range(N):
        var a = Float64(feature[i] - proto0[i])
        var b = Float64(feature[i] - proto1[i])
        d0 += a * a
        d1 += b * b
    return 0 if d0 <= d1 else 1

def direct_memory(condition: Int, sample_id: Int) -> List[Float32]:
    """Two leaky sensor-state units, with no connectome, as a simple baseline."""
    var state = zeros(2)
    for step in range(STEPS):
        state[0] *= 0.85
        state[1] *= 0.85
        var symbol = -1
        if step == 0:
            symbol = 0 if condition == 0 else 1
        elif step == 2:
            symbol = 1 if condition == 0 else 0
        if symbol >= 0:
            var count = 0
            for k in range(96):
                var pattern_hash = k * 7919 + sample_id * 104729 + k * sample_id * 37
                if pattern_hash % 100 >= 20:
                    count += 1
            state[symbol] += 0.08 * Float32(count) / 96.0
    return state^

def predict2(feature: List[Float32], proto0: List[Float32], proto1: List[Float32]) -> Int:
    var d0: Float64 = 0.0
    var d1: Float64 = 0.0
    for i in range(2):
        var a = Float64(feature[i] - proto0[i])
        var b = Float64(feature[i] - proto1[i])
        d0 += a * a
        d1 += b * b
    return 0 if d0 <= d1 else 1

def main() raises:
    var base = "../data/"
    var fr = open(base + "row_ptr.bin", "r")
    var row_bytes = fr.read_bytes()
    fr.close()
    var fc = open(base + "col_idx.bin", "r")
    var col_bytes = fc.read_bytes()
    fc.close()
    var fw = open(base + "weights.bin", "r")
    var weight_bytes = fw.read_bytes()
    fw.close()
    var row_ptr = row_bytes.unsafe_ptr().unsafe_bitcast[Int32]()
    var col_idx = col_bytes.unsafe_ptr().unsafe_bitcast[Int32]()
    var weights = weight_bytes.unsafe_ptr().unsafe_bitcast[Float32]()

    var proto0 = zeros(N)
    var proto1 = zeros(N)
    var shuffled0 = zeros(N)
    var shuffled1 = zeros(N)
    var direct0 = zeros(2)
    var direct1 = zeros(2)
    # Balanced training-label permutation, independent of true class parity.
    var permuted_labels = zeros(TRAIN)
    for s in range(TRAIN):
        permuted_labels[s] = 0.0 if ((s * 7 + 3) % TRAIN) < (TRAIN // 2) else 1.0
        var true_class = s % 2
        var sample_id = s // 2
        var feature = run_sample(row_ptr, col_idx, weights, true_class, sample_id)
        var direct = direct_memory(true_class, sample_id)
        var label = Int(permuted_labels[s])
        if true_class == 0:
            for i in range(2):
                direct0[i] += direct[i] * 0.0625
            for i in range(N):
                proto0[i] += feature[i] * 0.0625
        else:
            for i in range(2):
                direct1[i] += direct[i] * 0.0625
            for i in range(N):
                proto1[i] += feature[i] * 0.0625
        if label == 0:
            for i in range(N):
                shuffled0[i] += feature[i] * 0.0625
        else:
            for i in range(N):
                shuffled1[i] += feature[i] * 0.0625

    var correct = 0
    var shuffled_correct = 0
    var direct_correct = 0
    for s in range(TEST):
        var true_class = s % 2
        var sample_id = 100 + s // 2
        var feature = run_sample(row_ptr, col_idx, weights, true_class, sample_id)
        var direct = direct_memory(true_class, sample_id)
        if predict(feature, proto0, proto1) == true_class:
            correct += 1
        if predict(feature, shuffled0, shuffled1) == true_class:
            shuffled_correct += 1
        if predict2(direct, direct0, direct1) == true_class:
            direct_correct += 1

    print("probe=connectome_fixed_readout")
    print("graph_neurons=", N, " steps=", STEPS)
    print("train_per_class=", TRAIN // 2, " test_per_class=", TEST // 2)
    print("task=classify_AB_vs_BA_from_final_state_after_input_ends")
    print("input_mapping=synthetic_arbitrary_indices_not_sensory_labels")
    print("final_external_input=zero_for_both_classes")
    print("centroid_readout_accuracy=", correct, "/", TEST)
    print("two_unit_direct_memory_control=", direct_correct, "/", TEST)
    print("balanced_permuted_label_control=", shuffled_correct, "/", TEST)

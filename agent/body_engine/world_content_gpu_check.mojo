"""GPU plume sampler gate for the seed-0 scenario descriptor; formula source is agent/maps.py:104-120."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major, TensorLayout

from world_content import plume_concentration


def sample_kernel[LT: TensorLayout, LO: TensorLayout](
    xs: TileTensor[DType.float32, LT, MutAnyOrigin],
    ys: TileTensor[DType.float32, LT, MutAnyOrigin],
    output: TileTensor[DType.float32, LO, MutAnyOrigin],
    n: Int32,
    sx_food: Float32, sy_food: Float32, food_present: Int32, food_remaining: Float32,
    sx_decoy: Float32, sy_decoy: Float32, decoy_present: Int32, decoy_remaining: Float32,
):
    comptime assert xs.flat_rank == 1
    comptime assert ys.flat_rank == 1
    comptime assert output.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        output[2 * i] = plume_concentration(xs[i], ys[i], sx_food, sy_food,
                                            food_present, food_remaining)
        output[2 * i + 1] = plume_concentration(xs[i], ys[i], sx_decoy, sy_decoy,
                                                decoy_present, decoy_remaining)


def main() raises:
    comptime N = 4
    comptime B = 128
    var ctx = DeviceContext(0)
    var xs_h = ctx.enqueue_create_host_buffer[DType.float32](N)
    var ys_h = ctx.enqueue_create_host_buffer[DType.float32](N)
    var out_h = ctx.enqueue_create_host_buffer[DType.float32](2 * N)
    var xs_d = ctx.enqueue_create_buffer[DType.float32](N)
    var ys_d = ctx.enqueue_create_buffer[DType.float32](N)
    var out_d = ctx.enqueue_create_buffer[DType.float32](2 * N)
    var xs = TileTensor(xs_h, row_major[N]())
    var ys = TileTensor(ys_h, row_major[N]())
    for i in range(N):
        xs[i] = Float32((i + 1) * 250)
        ys[i] = Float32((i - 1) * 125)
    ctx.enqueue_copy(dst_buf=xs_d, src_buf=xs_h)
    ctx.enqueue_copy(dst_buf=ys_d, src_buf=ys_h)
    var k = ctx.compile_function[sample_kernel[type_of(row_major[N]()), type_of(row_major[2 * N]())]]()
    ctx.enqueue_function(k, TileTensor(xs_d, row_major[N]()), TileTensor(ys_d, row_major[N]()),
                         TileTensor(out_d, row_major[2 * N]()), Int32(N),
                         Float32(305.47847), Float32(340.79147), Int32(1), Float32(50),
                         Float32(281.63895), Float32(-369.33890), Int32(1), Float32(50),
                         grid_dim=1, block_dim=B)
    ctx.enqueue_copy(dst_buf=out_h, src_buf=out_d)
    ctx.synchronize()
    var out = TileTensor(out_h, row_major[2 * N]())
    for i in range(2 * N):
        print(out[i])

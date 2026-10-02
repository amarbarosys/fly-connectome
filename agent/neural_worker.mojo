"""Resident two-state FlyWire GPU reservoir, bounded sensory drive and DN readback.

Experimental interface: dynamics are the existing simplified LIF model. The
readout learns an engineered motor interface, not recovered biological behavior.
"""
from std.python import Python
from std.collections import List, InlineArray
from std.math import fma
from std.memory import unsafe_memset_zero
from std.os import getenv
from std.sys import get_defined_int, stderr
from std.time import perf_counter_ns
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime N = 139255
comptime E = 10572269
comptime C = get_defined_int["FLY_NEURAL_FLIES", 2]()
comptime D = 1303
comptime F = D * 2
comptime BLOCK = 128
comptime U = get_defined_int["FLY_NEURAL_UNROLL", 16]()
comptime ORDERED = get_defined_int["FLY_NEURAL_ORDERED", 0]()
comptime GRID = (N + BLOCK - 1) // BLOCK
comptime LR = row_major[N + 1]()
comptime LE = row_major[E]()
comptime LN = row_major[N]()
comptime LS = row_major[C * N]()
comptime LP = row_major[N * 8]()
comptime LI = row_major[C * 8]()
comptime LD = row_major[D]()
comptime LF = row_major[C * F]()


def update(
    rows: TileTensor[DType.int32, type_of(LR), MutAnyOrigin],
    cols: TileTensor[DType.int32, type_of(LE), MutAnyOrigin],
    weights: TileTensor[DType.float32, type_of(LE), MutAnyOrigin],
    ports: TileTensor[DType.uint8, type_of(LP), MutAnyOrigin],
    inputs: TileTensor[DType.float32, type_of(LI), MutAnyOrigin],
    direct: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    order: TileTensor[DType.int32, type_of(LN), MutAnyOrigin],
    spikes: TileTensor[DType.uint16, type_of(LN), MutAnyOrigin],
    next_spikes: TileTensor[DType.uint16, type_of(LN), MutAnyOrigin],
    voltage: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    trace: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    ever: TileTensor[DType.uint16, type_of(LN), MutAnyOrigin],
):
    """One thread per neuron advances every fly; bit `fly` of a spike word is that fly's spike."""
    var slot = block_idx.x * block_dim.x + thread_idx.x
    if slot >= N:
        return
    var neuron = Int(order[slot])
    var total = InlineArray[Float32, C](fill=Float32(0.0))
    var edge = Int(rows[neuron])
    var end = Int(rows[neuron + 1])
    # U edges per pass so their loads overlap; each fly's sum keeps edge order.
    while edge < end:
        var source = InlineArray[Int, U](uninitialized=True)
        var weight = InlineArray[Float32, U](uninitialized=True)
        comptime for k in range(U):
            var index = min(edge + k, end - 1)
            source[k] = Int(spikes[Int(cols[index])])
            weight[k] = weights[index]
        comptime for k in range(U):
            if edge + k < end:
                comptime for fly in range(C):
                    if (source[k] >> fly) & 1 != 0:
                        total[fly] += weight[k]
        edge += U
    # Load every fly's state before the first store so the loads overlap too.
    var drive = InlineArray[Float32, C](fill=Float32(0.0))
    var held = InlineArray[Float32, C](uninitialized=True)
    var faded = InlineArray[Float32, C](uninitialized=True)
    var seen = Int(ever[neuron])
    comptime for fly in range(C):
        # Explicit fma pins the contraction the per-fly kernel compiled to.
        comptime for channel in range(8):
            var gain: Float32 = inputs[fly * 8 + channel]
            drive[fly] = fma(Float32(ports[neuron * 8 + channel]), gain, drive[fly])
        drive[fly] += direct[fly * N + neuron]
        held[fly] = voltage[fly * N + neuron]
        faded[fly] = trace[fly * N + neuron]
    var fired = 0
    comptime for fly in range(C):
        var value = fma(Float32(0.65), drive[fly], fma(Float32(0.85), held[fly], 0.08 * total[fly]))
        var spike: Float32 = 1.0 if value >= 1.0 else 0.0
        voltage[fly * N + neuron] = 0.0 if spike > 0.0 else (value if value >= -5.0 else -5.0)
        trace[fly * N + neuron] = fma(Float32(0.85), faded[fly], 0.15 * spike)
        if spike > 0.0:
            fired |= 1 << fly
    next_spikes[neuron] = UInt16(fired)
    ever[neuron] = UInt16(seen | fired)


def expand(
    spikes: TileTensor[DType.uint16, type_of(LN), MutAnyOrigin],
    ever: TileTensor[DType.uint16, type_of(LN), MutAnyOrigin],
    spike_values: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    ever_values: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
):
    """Spike and recruitment words to the per-fly float arrays the host bridge reads."""
    var neuron = block_idx.x * block_dim.x + thread_idx.x
    if neuron >= N:
        return
    var fired = Int(spikes[neuron])
    var seen = Int(ever[neuron])
    comptime for fly in range(C):
        spike_values[fly * N + neuron] = Float32((fired >> fly) & 1)
        ever_values[fly * N + neuron] = Float32((seen >> fly) & 1)


def pack(
    spike_values: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    ever_values: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    spikes: TileTensor[DType.uint16, type_of(LN), MutAnyOrigin],
    ever: TileTensor[DType.uint16, type_of(LN), MutAnyOrigin],
):
    """Inverse of expand, for imported state."""
    var neuron = block_idx.x * block_dim.x + thread_idx.x
    if neuron >= N:
        return
    var fired = 0
    var seen = 0
    comptime for fly in range(C):
        if spike_values[fly * N + neuron] > 0:
            fired |= 1 << fly
        if ever_values[fly * N + neuron] > 0:
            seen |= 1 << fly
    spikes[neuron] = UInt16(fired)
    ever[neuron] = UInt16(seen)


def gather(
    voltage: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    trace: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    descending: TileTensor[DType.int32, type_of(LD), MutAnyOrigin],
    features: TileTensor[DType.float32, type_of(LF), MutAnyOrigin],
):
    var index = block_idx.x * block_dim.x + thread_idx.x
    var condition = block_idx.y
    if index < D:
        var neuron = condition * N + Int(descending[index])
        features[condition * F + index] = voltage[neuron]
        features[condition * F + D + index] = trace[neuron]


def main() raises:
    comptime assert 1 <= C <= 10, "neural worker supports one through ten independent flies"
    Python.add_to_path(getenv("FLY_PROJECT_ROOT", "/home/<user>/Projects/science/fly-connectome"))
    var bridge = Python.import_module("agent.neural_io")
    var arrays = Python.import_module("agent.neural_connectome")
    var assets = getenv("FLY_NEURAL_ASSETS", "/home/<user>/Projects/science/fly-connectome/.work/neural-flies/assets")
    var data_dir = "/home/<user>/Projects/science/fly-connectome/data/"
    # CPU validation uses the same roster-bound loader as the independent oracle.
    _ = arrays.graph_arrays()
    var row_data = open(data_dir + "row_ptr.bin", "r").read_bytes()
    var col_data = open(data_dir + "col_idx.bin", "r").read_bytes()
    var weight_data = open(data_dir + "weights.bin", "r").read_bytes()
    var port_data = open(assets + "/ports.bin", "r").read_bytes()
    var descending_data = open(assets + "/descending.bin", "r").read_bytes()
    if len(port_data) != N * 8 or len(descending_data) != D * 4:
        raise Error("invalid neural asset dimensions")
    if getenv("FLY_NEURAL_DRY") == "1":
        bridge.ready(F, C, True)
        return
    # Stable counting sort of neurons by descending in-degree.
    var row_ptr = row_data.unsafe_ptr().unsafe_bitcast[Int32]()
    var max_degree = 0
    for i in range(N):
        max_degree = max(max_degree, Int(row_ptr[unsafe_offset=i + 1]) - Int(row_ptr[unsafe_offset=i]))
    var bucket = List[Int](capacity=max_degree + 2)
    for _ in range(max_degree + 2):
        bucket.append(0)
    for i in range(N):
        bucket[max_degree - (Int(row_ptr[unsafe_offset=i + 1]) - Int(row_ptr[unsafe_offset=i])) + 1] += 1
    for key in range(max_degree + 1):
        bucket[key + 1] += bucket[key]
    var host_order = List[Int32](capacity=N)
    for _ in range(N):
        host_order.append(0)
    for i in range(N):
        var key = max_degree - (Int(row_ptr[unsafe_offset=i + 1]) - Int(row_ptr[unsafe_offset=i]))
        host_order[bucket[key]] = Int32(i)
        bucket[key] += 1
    if ORDERED == 0:
        for i in range(N):
            host_order[i] = Int32(i)
    var ctx = DeviceContext(0)
    var dynamics = ctx.compile_function[update]()
    var unpack = ctx.compile_function[expand]()
    var repack = ctx.compile_function[pack]()
    var readout = ctx.compile_function[gather]()
    var row_buffer = ctx.enqueue_create_buffer[DType.int32](N + 1)
    var col_buffer = ctx.enqueue_create_buffer[DType.int32](E)
    var weight_buffer = ctx.enqueue_create_buffer[DType.float32](E)
    var port_buffer = ctx.enqueue_create_buffer[DType.uint8](N * 8)
    var descending_buffer = ctx.enqueue_create_buffer[DType.int32](D)
    var order_buffer = ctx.enqueue_create_buffer[DType.int32](N)
    var input_buffer = ctx.enqueue_create_buffer[DType.float32](C * 8)
    var direct_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    # Spikes and recruitment live as one bit per fly; the float copies exist only for host transfer.
    var x_buffer = ctx.enqueue_create_buffer[DType.uint16](N)
    var y_buffer = ctx.enqueue_create_buffer[DType.uint16](N)
    var ever_buffer = ctx.enqueue_create_buffer[DType.uint16](N)
    var x_float_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var ever_float_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var v_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var trace_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var feature_buffer = ctx.enqueue_create_buffer[DType.float32](C * F)
    ctx.enqueue_copy(row_buffer, row_data.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(col_buffer, col_data.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(weight_buffer, weight_data.unsafe_ptr().unsafe_bitcast[Float32]())
    if getenv("FLY_NEURAL_DISCONNECT") == "1":
        weight_buffer.enqueue_fill(0.0)
    ctx.enqueue_copy(port_buffer, port_data.unsafe_ptr())
    ctx.enqueue_copy(descending_buffer, descending_data.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(order_buffer, host_order.unsafe_ptr())
    var row_tensor = TileTensor(row_buffer, LR)
    var col_tensor = TileTensor(col_buffer, LE)
    var weight_tensor = TileTensor(weight_buffer, LE)
    var port_tensor = TileTensor(port_buffer, LP)
    var input_tensor = TileTensor(input_buffer, LI)
    var direct_tensor = TileTensor(direct_buffer, LS)
    var order_tensor = TileTensor(order_buffer, LN)
    var x_tensor = TileTensor(x_buffer, LN)
    var y_tensor = TileTensor(y_buffer, LN)
    var ever_tensor = TileTensor(ever_buffer, LN)
    var x_float_tensor = TileTensor(x_float_buffer, LS)
    var ever_float_tensor = TileTensor(ever_float_buffer, LS)
    var v_tensor = TileTensor(v_buffer, LS)
    var trace_tensor = TileTensor(trace_buffer, LS)
    var descending_tensor = TileTensor(descending_buffer, LD)
    var feature_tensor = TileTensor(feature_buffer, LF)
    var host_direct = List[Float32](capacity=C * N)
    for _ in range(C * N):
        host_direct.append(0.0)
    var host_inputs = List[Float32](capacity=C * 8)
    var host_features = List[Float32](capacity=C * F)
    var host_spikes = List[Float32](capacity=C * N)
    var host_ever = List[Float32](capacity=C * N)
    var host_voltage = List[Float32](capacity=C * N)
    var host_trace = List[Float32](capacity=C * N)
    for _ in range(C * 8):
        host_inputs.append(0.0)
    for _ in range(C * F):
        host_features.append(0.0)
    for _ in range(C * N):
        host_spikes.append(0.0)
        host_ever.append(0.0)
        host_voltage.append(0.0)
        host_trace.append(0.0)
    x_buffer.enqueue_fill(0)
    y_buffer.enqueue_fill(0)
    ever_buffer.enqueue_fill(0)
    v_buffer.enqueue_fill(0.0)
    trace_buffer.enqueue_fill(0.0)
    ctx.synchronize()
    # FLY_NEURAL_TIMING=1 adds a synchronize per phase and one stderr line per exchange (microseconds).
    var timing = getenv("FLY_NEURAL_TIMING") == "1"
    bridge.ready(F, C, True)
    while True:
        var request = bridge.receive(C)
        var op = String(py=request["op"])
        if op == "quit":
            break
        if op == "reset":
            x_buffer.enqueue_fill(0)
            y_buffer.enqueue_fill(0)
            ever_buffer.enqueue_fill(0)
            v_buffer.enqueue_fill(0.0)
            trace_buffer.enqueue_fill(0.0)
            unsafe_memset_zero(host_ever.unsafe_ptr(), C * N)
            ctx.synchronize()
            bridge.reset_reply()
            continue
        if op == "export_state" or op == "import_state":
            if op == "export_state":
                ctx.enqueue_function(unpack, x_tensor, ever_tensor, x_float_tensor, ever_float_tensor,
                    grid_dim=GRID, block_dim=BLOCK)
                ctx.enqueue_copy(host_voltage.unsafe_ptr(), v_buffer)
                ctx.enqueue_copy(host_spikes.unsafe_ptr(), x_float_buffer)
                ctx.enqueue_copy(host_trace.unsafe_ptr(), trace_buffer)
                ctx.enqueue_copy(host_ever.unsafe_ptr(), ever_float_buffer)
                ctx.synchronize()
            bridge.state_buffers(request, Int(host_voltage.unsafe_ptr()), Int(host_spikes.unsafe_ptr()),
                Int(host_trace.unsafe_ptr()), Int(host_ever.unsafe_ptr()), C)
            if op == "import_state":
                ctx.enqueue_copy(v_buffer, host_voltage.unsafe_ptr())
                ctx.enqueue_copy(x_float_buffer, host_spikes.unsafe_ptr())
                ctx.enqueue_copy(trace_buffer, host_trace.unsafe_ptr())
                ctx.enqueue_copy(ever_float_buffer, host_ever.unsafe_ptr())
                ctx.enqueue_function(repack, x_float_tensor, ever_float_tensor, x_tensor, ever_tensor,
                    grid_dim=GRID, block_dim=BLOCK)
                y_buffer.enqueue_fill(0)
                ctx.synchronize()
            bridge.state_reply()
            continue
        var t0 = perf_counter_ns()
        for i in range(C * 8):
            host_inputs[i] = Float32(Float64(py=request["inputs"][i]))
        unsafe_memset_zero(host_direct.unsafe_ptr(), C * N)
        bridge.apply_visual(request, Int(host_direct.unsafe_ptr()), C)
        for i in range(Int(py=request["direct"].__len__())):
            var target = Int(py=request["direct"][i])
            if host_ever[target] == 0.0:
                host_direct[target] += 1.0
        var t1 = perf_counter_ns()
        ctx.enqueue_copy(direct_buffer, host_direct.unsafe_ptr())
        ctx.enqueue_copy(input_buffer, host_inputs.unsafe_ptr())
        if timing:
            ctx.synchronize()
        var t2 = perf_counter_ns()
        for tick in range(4):
            if tick % 2 == 0:
                ctx.enqueue_function(dynamics, row_tensor, col_tensor, weight_tensor,
                    port_tensor, input_tensor, direct_tensor, order_tensor, x_tensor, y_tensor, v_tensor,
                    trace_tensor, ever_tensor, grid_dim=GRID, block_dim=BLOCK)
            else:
                ctx.enqueue_function(dynamics, row_tensor, col_tensor, weight_tensor,
                    port_tensor, input_tensor, direct_tensor, order_tensor, y_tensor, x_tensor, v_tensor,
                    trace_tensor, ever_tensor, grid_dim=GRID, block_dim=BLOCK)
        if timing:
            ctx.synchronize()
        var t3 = perf_counter_ns()
        ctx.enqueue_function(unpack, x_tensor, ever_tensor, x_float_tensor, ever_float_tensor,
            grid_dim=GRID, block_dim=BLOCK)
        ctx.enqueue_function(readout, v_tensor, trace_tensor, descending_tensor,
            feature_tensor, grid_dim=((D + BLOCK - 1) // BLOCK, C), block_dim=BLOCK)
        ctx.enqueue_copy(host_features.unsafe_ptr(), feature_buffer)
        ctx.enqueue_copy(host_spikes.unsafe_ptr(), x_float_buffer)
        ctx.enqueue_copy(host_ever.unsafe_ptr(), ever_float_buffer)
        ctx.synchronize()
        var t4 = perf_counter_ns()
        bridge.reply_buffers(Int(host_features.unsafe_ptr()), Int(host_spikes.unsafe_ptr()),
            Int(host_ever.unsafe_ptr()), C, F, N, Int(py=request["after"]))
        if timing:
            print("T", (t1 - t0) // 1000, (t2 - t1) // 1000, (t3 - t2) // 1000, (t4 - t3) // 1000,
                (perf_counter_ns() - t4) // 1000, file=stderr)

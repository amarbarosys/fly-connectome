"""Resident two-state FlyWire GPU reservoir, bounded sensory drive and DN readback.

Experimental interface: dynamics are the existing simplified LIF model. The
readout learns an engineered motor interface, not recovered biological behavior.
"""
from std.python import Python
from std.collections import List
from std.os import getenv
from std.sys import get_defined_int
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime N = 139255
comptime E = 10572269
comptime C = get_defined_int["FLY_NEURAL_FLIES", 2]()
comptime D = 1303
comptime F = D * 2
comptime BLOCK = 128
comptime GRID = (N + BLOCK - 1) // BLOCK
comptime LR = row_major[N + 1]()
comptime LE = row_major[E]()
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
    spikes: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    next_spikes: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    voltage: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    trace: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
    ever: TileTensor[DType.float32, type_of(LS), MutAnyOrigin],
):
    var neuron = block_idx.x * block_dim.x + thread_idx.x
    var condition = block_idx.y
    if neuron >= N:
        return
    var base = condition * N
    var total: Float32 = 0.0
    for edge in range(Int(rows[neuron]), Int(rows[neuron + 1])):
        var source_spike = spikes[base + Int(cols[edge])]
        if source_spike > 0:
            total += weights[edge] * source_spike
    var drive: Float32 = 0.0
    comptime for channel in range(8):
        drive += Float32(ports[neuron * 8 + channel]) * inputs[condition * 8 + channel]
    var value = 0.85 * voltage[base + neuron] + 0.08 * total + 0.65 * (drive + direct[base + neuron])
    var spike: Float32 = 1.0 if value >= 1.0 else 0.0
    next_spikes[base + neuron] = spike
    voltage[base + neuron] = 0.0 if spike > 0.0 else (value if value >= -5.0 else -5.0)
    trace[base + neuron] = 0.85 * trace[base + neuron] + 0.15 * spike
    if spike > 0.0:
        ever[base + neuron] = 1.0


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
    var ctx = DeviceContext(0)
    var dynamics = ctx.compile_function[update]()
    var readout = ctx.compile_function[gather]()
    var row_buffer = ctx.enqueue_create_buffer[DType.int32](N + 1)
    var col_buffer = ctx.enqueue_create_buffer[DType.int32](E)
    var weight_buffer = ctx.enqueue_create_buffer[DType.float32](E)
    var port_buffer = ctx.enqueue_create_buffer[DType.uint8](N * 8)
    var descending_buffer = ctx.enqueue_create_buffer[DType.int32](D)
    var input_buffer = ctx.enqueue_create_buffer[DType.float32](C * 8)
    var direct_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var x_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var y_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var v_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var trace_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var ever_buffer = ctx.enqueue_create_buffer[DType.float32](C * N)
    var feature_buffer = ctx.enqueue_create_buffer[DType.float32](C * F)
    ctx.enqueue_copy(row_buffer, row_data.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(col_buffer, col_data.unsafe_ptr().unsafe_bitcast[Int32]())
    ctx.enqueue_copy(weight_buffer, weight_data.unsafe_ptr().unsafe_bitcast[Float32]())
    if getenv("FLY_NEURAL_DISCONNECT") == "1":
        weight_buffer.enqueue_fill(0.0)
    ctx.enqueue_copy(port_buffer, port_data.unsafe_ptr())
    ctx.enqueue_copy(descending_buffer, descending_data.unsafe_ptr().unsafe_bitcast[Int32]())
    var row_tensor = TileTensor(row_buffer, LR)
    var col_tensor = TileTensor(col_buffer, LE)
    var weight_tensor = TileTensor(weight_buffer, LE)
    var port_tensor = TileTensor(port_buffer, LP)
    var input_tensor = TileTensor(input_buffer, LI)
    var direct_tensor = TileTensor(direct_buffer, LS)
    var x_tensor = TileTensor(x_buffer, LS)
    var y_tensor = TileTensor(y_buffer, LS)
    var v_tensor = TileTensor(v_buffer, LS)
    var trace_tensor = TileTensor(trace_buffer, LS)
    var ever_tensor = TileTensor(ever_buffer, LS)
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
    x_buffer.enqueue_fill(0.0)
    y_buffer.enqueue_fill(0.0)
    v_buffer.enqueue_fill(0.0)
    trace_buffer.enqueue_fill(0.0)
    ever_buffer.enqueue_fill(0.0)
    ctx.synchronize()
    bridge.ready(F, C, True)
    while True:
        var request = bridge.receive(C)
        var op = String(py=request["op"])
        if op == "quit":
            break
        if op == "reset":
            x_buffer.enqueue_fill(0.0)
            y_buffer.enqueue_fill(0.0)
            v_buffer.enqueue_fill(0.0)
            trace_buffer.enqueue_fill(0.0)
            ever_buffer.enqueue_fill(0.0)
            for i in range(C * N):
                host_ever[i] = 0.0
            ctx.synchronize()
            bridge.reset_reply()
            continue
        if op == "export_state" or op == "import_state":
            if op == "export_state":
                ctx.enqueue_copy(host_voltage.unsafe_ptr(), v_buffer)
                ctx.enqueue_copy(host_spikes.unsafe_ptr(), x_buffer)
                ctx.enqueue_copy(host_trace.unsafe_ptr(), trace_buffer)
                ctx.enqueue_copy(host_ever.unsafe_ptr(), ever_buffer)
                ctx.synchronize()
            bridge.state_buffers(request, Int(host_voltage.unsafe_ptr()), Int(host_spikes.unsafe_ptr()),
                Int(host_trace.unsafe_ptr()), Int(host_ever.unsafe_ptr()), C)
            if op == "import_state":
                ctx.enqueue_copy(v_buffer, host_voltage.unsafe_ptr())
                ctx.enqueue_copy(x_buffer, host_spikes.unsafe_ptr())
                ctx.enqueue_copy(trace_buffer, host_trace.unsafe_ptr())
                ctx.enqueue_copy(ever_buffer, host_ever.unsafe_ptr())
                y_buffer.enqueue_fill(0.0)
                ctx.synchronize()
            bridge.state_reply()
            continue
        for i in range(C * 8):
            host_inputs[i] = Float32(Float64(py=request["inputs"][i]))
        for i in range(C * N):
            host_direct[i] = 0.0
        bridge.apply_visual(request, Int(host_direct.unsafe_ptr()), C)
        for i in range(Int(py=request["direct"].__len__())):
            var target = Int(py=request["direct"][i])
            if host_ever[target] == 0.0:
                host_direct[target] += 1.0
        ctx.enqueue_copy(direct_buffer, host_direct.unsafe_ptr())
        ctx.enqueue_copy(input_buffer, host_inputs.unsafe_ptr())
        for tick in range(4):
            if tick % 2 == 0:
                ctx.enqueue_function(dynamics, row_tensor, col_tensor, weight_tensor,
                    port_tensor, input_tensor, direct_tensor, x_tensor, y_tensor, v_tensor,
                    trace_tensor, ever_tensor, grid_dim=(GRID, C), block_dim=BLOCK)
            else:
                ctx.enqueue_function(dynamics, row_tensor, col_tensor, weight_tensor,
                    port_tensor, input_tensor, direct_tensor, y_tensor, x_tensor, v_tensor,
                    trace_tensor, ever_tensor, grid_dim=(GRID, C), block_dim=BLOCK)
        ctx.enqueue_function(readout, v_tensor, trace_tensor, descending_tensor,
            feature_tensor, grid_dim=((D + BLOCK - 1) // BLOCK, C), block_dim=BLOCK)
        ctx.enqueue_copy(host_features.unsafe_ptr(), feature_buffer)
        ctx.enqueue_copy(host_spikes.unsafe_ptr(), x_buffer)
        ctx.enqueue_copy(host_ever.unsafe_ptr(), ever_buffer)
        ctx.synchronize()
        bridge.reply_buffers(Int(host_features.unsafe_ptr()), Int(host_spikes.unsafe_ptr()),
            Int(host_ever.unsafe_ptr()), C, F, N, Int(py=request["after"]))

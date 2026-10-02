#!/usr/bin/env mojo
"""Mechanics-only latent-to-added-neuron probe; no task semantics claimed."""

from std.collections import List

comptime D = 2048
comptime UNITS = 128
comptime STEPS = 4

def zeros(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(0.0)
    return out^

def run_population(hidden: Pointer[Float32, _], mode: Int) -> List[Float32]:
    var voltage = zeros(UNITS)
    var spikes = zeros(UNITS)
    var next_spikes = zeros(UNITS)
    var spike_counts = zeros(UNITS)

    for _ in range(STEPS):
        for unit in range(UNITS):
            var drive: Float32 = 0.0
            for dim in range(D):
                var value: Float32 = 0.0
                if mode == 0:
                    value = hidden[unsafe_offset=dim]
                elif mode == 2:
                    value = hidden[unsafe_offset=(dim * 37) % D]
                var projection_sign: Float32 = 1.0
                var projection_hash = (unit * 1103515245 + dim * 12345 + unit * dim * 2654435761) % 2147483647
                if projection_hash % 2 == 0:
                    projection_sign = -1.0
                drive += value * projection_sign * 0.0220970869

            var recurrent = spikes[(unit + UNITS - 1) % UNITS] * 0.15
            var membrane = voltage[unit] * 0.85 + drive * 0.50 + recurrent
            if membrane >= 1.0:
                next_spikes[unit] = 1.0
                voltage[unit] = 0.0
            else:
                next_spikes[unit] = 0.0
                if membrane < -5.0:
                    voltage[unit] = -5.0
                else:
                    voltage[unit] = membrane

        for unit in range(UNITS):
            spikes[unit] = next_spikes[unit]
            spike_counts[unit] += next_spikes[unit]

    return spike_counts^

def main() raises:
    var path = "../../../AiTrainer/datasets/mojo-baro/hidden-vectors/regescore/regescore-source-row.f32"
    var f = open(path, "r")
    var raw = f.read_bytes()
    f.close()
    if len(raw) != D * 4:
        raise Error("latent vector byte length does not match expected width 2048")
    var hidden = raw.unsafe_ptr().unsafe_bitcast[Float32]()

    var real = run_population(hidden, 0)
    var zero = run_population(hidden, 1)
    var shuffled = run_population(hidden, 2)
    var real_active = 0
    var zero_active = 0
    var shuffled_active = 0
    var real_spikes = 0
    var shuffled_spikes = 0
    var overlap_shuffled = 0
    var projection_min: Float32 = 1.0e30
    var projection_max: Float32 = -1.0e30
    for unit in range(UNITS):
        var projection: Float32 = 0.0
        for dim in range(D):
            var sign: Float32 = 1.0
            var projection_hash = (unit * 1103515245 + dim * 12345 + unit * dim * 2654435761) % 2147483647
            if projection_hash % 2 == 0:
                sign = -1.0
            projection += hidden[unsafe_offset=dim] * sign * 0.0220970869
        if projection < projection_min:
            projection_min = projection
        if projection > projection_max:
            projection_max = projection
        if real[unit] > 0.0:
            real_active += 1
            real_spikes += Int(real[unit])
        if zero[unit] > 0.0:
            zero_active += 1
        if shuffled[unit] > 0.0:
            shuffled_active += 1
            shuffled_spikes += Int(shuffled[unit])
        if real[unit] > 0.0 and shuffled[unit] > 0.0:
            overlap_shuffled += 1

    print("probe=latent_to_added_LIF_population_mechanics")
    print("latent_width=", D, " added_units=", UNITS, " steps=", STEPS)
    print("projection_range=", projection_min, "..", projection_max)
    print("latent_arm_units_spiking=", real_active)
    print("zero_control_units_spiking=", zero_active)
    print("shuffled_control_units_spiking=", shuffled_active)
    print("latent_total_spikes=", real_spikes)
    print("shuffled_total_spikes=", shuffled_spikes)
    print("latent_minus_shuffled_total=", real_spikes - shuffled_spikes)
    print("latent_shuffled_active_unit_overlap=", overlap_shuffled)
    print("task_score=NOT_MEASURED")
    print("biology=NOT_ESTABLISHED")

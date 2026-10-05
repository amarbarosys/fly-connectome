"""Compact game-state transitions for GAME_LAYER items 4-7.

The engine hook should call these from the existing actuator/step work once
the owner approves the integration point. The CPU contract and event semantics
live in game_events.py. No per-step host update is required.

Port basis: MuJoCo 3.9.0 `engine_forward.c:312` (`mj_fwdActuation`) supplies
the physical actuator force used with velocity to account for work. Game bars
and tagged scenario events are game-layer extensions, not MuJoCo physics ports.
"""

@fieldwise_init
struct GameState(Copyable, Movable):
    var energy: Float32
    var health: Float32
    var stamina: Float32
    # 0.0 or 1.0 keeps the row exactly four float32 values (16 B).
    var dead: Float32


def clamp01(x: Float32) -> Float32:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def nonnegative(x: Float32) -> Float32:
    if x < 0.0:
        return 0.0
    return x


def work_drain(work: Float32, basal: Float32, rate: Float32) -> Float32:
    """Convert measured actuator work into game drain, independent of command magnitude."""
    return nonnegative(basal) + nonnegative(work) * nonnegative(rate)


def advance_state(
    state: GameState,
    work: Float32,
    damage_event: Bool,
    resting: Bool,
    resource_granted: Bool,
    basal_drain: Float32,
    work_rate: Float32,
    damage_amount: Float32,
    regen_rate: Float32,
    rest_rate: Float32,
    resource_energy: Float32,
) -> GameState:
    """Advance one animal row; damage_event is one deduplicated tagged event."""
    if state.dead != 0.0:
        return GameState(state.energy, state.health, state.stamina, state.dead)
    var energy = clamp01(state.energy - work_drain(work, basal_drain, work_rate))
    var health = clamp01(state.health + nonnegative(regen_rate))
    var stamina = clamp01(state.stamina - nonnegative(work) * nonnegative(work_rate))
    if resting:
        stamina = clamp01(stamina + nonnegative(rest_rate))
    if damage_event:
        health = clamp01(health - nonnegative(damage_amount))
    if resource_granted:
        energy = clamp01(energy + nonnegative(resource_energy))
    var dead: Float32 = 0.0
    if energy <= 0.0 or health <= 0.0:
        dead = 1.0
    return GameState(energy, health, stamina, dead)


def actuator_scale(state: GameState, floor_scale: Float32) -> Float32:
    """Scale servo and adhesion effort; dead state explicitly yields zero effort."""
    if state.dead != 0.0:
        return 0.0
    var stamina = clamp01(state.stamina)
    var floor = clamp01(floor_scale)
    if stamina < floor:
        return floor
    return stamina


def main() raises:
    comptime assert 4 * 4 == 16, "GameState row is four float32 values"
    var alive = GameState(1.0, 1.0, 1.0, 0.0)
    var drained = advance_state(alive, 2.0, False, False, False, 0.0, 0.1,
                                0.0, 0.0, 0.1, 0.0)
    assert abs(drained.energy - 0.8) < 1e-6
    var fed = advance_state(drained, 0.0, False, False, True, 0.0, 0.1,
                            0.0, 0.0, 0.0, 0.25)
    assert abs(fed.energy - 1.0) < 1e-6
    assert actuator_scale(GameState(0.4, 1.0, 0.0, 0.0), 0.15) == 0.15
    var dead = advance_state(GameState(0.0, 1.0, 1.0, 1.0), 5.0, True, False,
                             False, 0.1, 0.2, 1.0, 0.0, 0.1, 0.0)
    assert dead.energy == 0.0 and dead.health == 1.0 and dead.dead == 1.0
    assert actuator_scale(GameState(0.0, 0.0, 0.0, 1.0), 0.15) == 0.0
    print("GAME_STATE PASS: work drain, 16 B row, rest-independent state and dead effort gate")

"""Live world edits for a running fly (docs/plans/CONTROL_PANEL.md phase 3).

One food patch lives in three places that must move together: the odour/taste model
(`Environment.food`, a FoodPatch), the arena's contact list (`arena.food_patches`, dicts the
mouth test reads) and the MuJoCo disc the eyes see (geom `food_<i>`). An edit is validated
whole before any of them changes, then applied to all three at one exchange boundary.
"""
import dataclasses

EDITABLE = ('x', 'y', 'amount')


def validate(edit, environment):
    if not isinstance(edit, dict): raise ValueError('edit must be an object')
    unknown = set(edit) - {'food_id', *EDITABLE}
    if unknown: raise ValueError(f'unknown edit fields: {sorted(unknown)}')
    ids = [p.id for p in environment.food]
    if edit.get('food_id') not in ids: raise ValueError(f'food_id must be one of {ids}')
    changes = {k: edit[k] for k in EDITABLE if k in edit}
    if not changes: raise ValueError(f'edit changes nothing; give any of {EDITABLE}')
    if not all(type(v) in (int, float) for v in changes.values()): raise ValueError('edit values must be numbers')
    index = ids.index(edit['food_id'])
    # FoodPatch.__post_init__ is the world's own rule for a legal patch (finite, amount >= 0).
    dataclasses.replace(environment.food[index], **{k: float(v) for k, v in changes.items()})
    return index, {k: float(v) for k, v in changes.items()}


def apply(edit, environment, arena):
    index, changes = validate(edit, environment)
    patch = environment.food[index]
    before = {k: getattr(patch, k) for k in changes}
    for k, v in changes.items():
        setattr(patch, k, v)
        if k in ('x', 'y'): arena.food_patches[index][k] = v
    if {'x', 'y'} & set(changes):
        model = arena.sim.mj_model
        geom = model.geom(f'food_{index}').id
        model.geom_pos[geom][:2] = (patch.x, patch.y)
    return {'food_id': patch.id, 'before': before, 'after': changes}

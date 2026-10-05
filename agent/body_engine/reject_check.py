"""implicitfast must refuse, at export, every model whose velocity derivatives (fluid, tendon) are not ported
(mjd_passive_vel fluid and tendon-damping terms in engine_derivative.c). Each case builds a tiny MJCF and requires
model.load to exit with 'FAIL export'. usage: python -m agent.body_engine.reject_check"""
import sys

import mujoco

from agent.body_engine.model import load

BASE = '<mujoco><option integrator="implicitfast"{opt}/><worldbody><body name="b"><freejoint/><geom size="1"/>' \
       '<body name="c"><joint name="j" axis="0 1 0"/><geom size="1"/></body></body></worldbody>{extra}</mujoco>'
CASES = {
    'viscosity': dict(opt=' viscosity="1e-3"', extra=''),
    'density': dict(opt=' density="1e-3"', extra=''),
    'tendon damping': dict(opt='', extra='<tendon><fixed name="t" damping="1"><joint joint="j" coef="1"/></fixed></tendon>'),
}
ACCEPT = {  # no velocity derivative in qDeriv: a spring-only tendon is exported as under Euler
    'tendon spring only': dict(opt='', extra='<tendon><fixed name="t" stiffness="1"><joint joint="j" coef="1"/></fixed></tendon>'),
}
bad = []
for name, kw in CASES.items():
    m = mujoco.MjModel.from_xml_string(BASE.format(**kw))
    try:
        load(m)
        bad.append(name)
        print(f'FAIL reject {name}: load accepted it under implicitfast')
    except SystemExit as e:
        ok = str(e).startswith('FAIL export: implicitfast needs')
        print(f'{"PASS" if ok else "FAIL"} reject {name}: {e}')
        bad += [] if ok else [name]
for name, kw in ACCEPT.items():
    try:
        load(mujoco.MjModel.from_xml_string(BASE.format(**kw)))
        print(f'PASS accept {name}')
    except SystemExit as e:
        print(f'FAIL accept {name}: {e}')
        bad.append(name)
sys.exit(f'FAIL reject: {bad}' if bad else 0)

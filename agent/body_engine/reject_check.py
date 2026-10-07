"""implicit and implicitfast must refuse, at export, every model whose velocity derivatives (fluid) are not ported
(mjd_passive_vel fluid terms in engine_derivative.c; the tendon-damping term is ported), under both integrators. Each case builds a tiny MJCF and requires
model.load to exit with 'FAIL export'. usage: python -m agent.body_engine.reject_check"""
import sys

import mujoco

from agent.body_engine.model import load

BASE = '<mujoco><option integrator="{integ}"{opt}/><worldbody><body name="b"><freejoint/><geom size="1"/>' \
       '<body name="c"><joint name="j" axis="0 1 0"/><geom size="1"/></body></body></worldbody>{extra}</mujoco>'
CASES = {
    'viscosity': dict(opt=' viscosity="1e-3"', extra=''),
    'density': dict(opt=' density="1e-3"', extra=''),
}
ACCEPT = {  # a spring-only tendon has no velocity derivative; tendon damping's is ported (check.sh implicit, 2026-10-07)
    'tendon spring only': dict(opt='', extra='<tendon><fixed name="t" stiffness="1"><joint joint="j" coef="1"/></fixed></tendon>'),
    'tendon damping': dict(opt='', extra='<tendon><fixed name="t" damping="1"><joint joint="j" coef="1"/></fixed></tendon>'),
}
bad = []
for integ in ('implicitfast', 'implicit'):
    for name, kw in CASES.items():
        m = mujoco.MjModel.from_xml_string(BASE.format(integ=integ, **kw))
        try:
            load(m)
            bad.append(f'{integ} {name}')
            print(f'FAIL reject {name}: load accepted it under {integ}')
        except SystemExit as e:
            ok = str(e).startswith('FAIL export: implicit and implicitfast need')
            print(f'{"PASS" if ok else "FAIL"} reject {integ} {name}: {e}')
            bad += [] if ok else [f'{integ} {name}']
    for name, kw in ACCEPT.items():
        try:
            load(mujoco.MjModel.from_xml_string(BASE.format(integ=integ, **kw)))
            print(f'PASS accept {integ} {name}')
        except SystemExit as e:
            print(f'FAIL accept {integ} {name}: {e}')
            bad.append(f'{integ} {name}')
sys.exit(f'FAIL reject: {bad}' if bad else 0)

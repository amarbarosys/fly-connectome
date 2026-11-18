"""Model desert ant body (Cataglyphis-like) as MJCF for the body engine; private note ANT unit A1.

Built like kinestride.models.spider (same units mm/g/s, dt 0.1 ms, convex meshes, plane-mesh contact pairs with the
fly's solref/solimp, position servos; its prism/ellipsoid/stand and servo constants are reused). Anatomy approximate
for Cataglyphis fortis, NOT measured from a specimen: head 1.6 mm, mesosoma 3.2 mm, petiole on a 2-hinge joint, gaster
2.8 mm; three leg pairs on the mesosoma, each coxa, femur, tibia, tarsus with long running legs; 5 hinges per leg
(coxa yaw + pitch, femur, tibia, tarsus). Zero joint angles are the standing pose, so servos at 0 hold it.
Not modelled: antennae, mandibles, the tarsal claws and adhesive pads.

usage: python -m agent.ant_body [OUT_DIR]   -> OUT_DIR/ant.xml + meshes; prints the stand test
"""
import sys
from pathlib import Path

import numpy as np

from kinestride.models.spider import DAMP, FMAX, KP, ellipsoid, prism, stand

OUT = Path('.work/ant-body')
SEG = [('coxa', 0.7, -20.0), ('femur', 2.4, 45.0), ('tibia', 2.4, -95.0), ('tarsus', 2.0, -35.0)]  # name, mm, pitch vs parent (deg, + up)
LEGS = [(1, 0.9, 50.0, 1.0, 0.22, 0.0), (2, 0.0, 0.0, 1.1, 0.2, 6.56), (3, -0.9, -45.0, 1.25, 0.2, 13.7)]  # pair, x on mesosoma, yaw, length scale, radius, femur pitch offset (deg; levels the feet)


def build(out=OUT, legs=LEGS, z=3.85):  # z: tarsi 1.4 mm below the root at 0 deg, start 0.05 mm above ground
    out.mkdir(parents=True, exist_ok=True)
    for part, ax in (('head', (0.8, 0.65, 0.55)), ('mesosoma', (1.6, 0.55, 0.55)), ('petiole', (0.3, 0.25, 0.3)), ('gaster', (1.4, 0.95, 0.85))):
        ellipsoid(out / f'{part}.stl', *ax)
    assets = [f'<mesh name="{p}" file="{p}.stl"/>' for p in ('head', 'mesosoma', 'petiole', 'gaster')]
    actuators, pairs = [], ['head', 'mesosoma', 'petiole', 'gaster']

    def leg(side, pair, x, yaw, scale, radius, lift):
        name = f'{"l" if side > 0 else "r"}{pair}'
        quat = (np.cos(side * np.radians(90 - yaw) / 2), np.sin(side * np.radians(90 - yaw) / 2))  # leg x points outward
        xml, close, r = [], [], radius
        for k, (seg, length, pitch) in enumerate(SEG):
            mesh, L = f'{name}_{seg}', length * scale
            prism(out / f'{mesh}.stl', L, r, r * 0.85)
            assets.append(f'<mesh name="{mesh}" file="{mesh}.stl"/>')
            pitch += lift if seg == 'femur' else 0.0
            q = (np.cos(np.radians(-pitch) / 2), np.sin(np.radians(-pitch) / 2))
            if k == 0:
                xml.append(f'<body name="{mesh}" pos="{x:.3f} {side * 0.4:.3f} -0.35" quat="{quat[0]:.6f} 0 0 {quat[1]:.6f}">'
                           f'<body name="{mesh}_p" quat="{q[0]:.6f} 0 {q[1]:.6f} 0">')
                joints, close = [(f'{name}_coxa_yaw', '0 0 1'), (f'{name}_coxa_pitch', '0 1 0')], ['</body>']
            else:
                xml.append(f'<body name="{mesh}" pos="{SEG[k - 1][1] * scale:.3f} 0 0" quat="{q[0]:.6f} 0 {q[1]:.6f} 0">')
                joints = [(f'{name}_{seg}', '0 1 0')]
            for j, axis in joints:
                xml.append(f'<joint name="{j}" axis="{axis}" range="-60 60"/>')
                actuators.append(f'<position name="{j}" joint="{j}" kp="{KP}" forcelimited="true" forcerange="{-FMAX} {FMAX}"/>')
            xml.append(f'<geom name="{mesh}" type="mesh" mesh="{mesh}"/>')
            pairs.append(mesh)
            close.append('</body>')
            r *= 0.85
        return '\n'.join(xml + close[::-1])

    legs = '\n'.join(leg(s, *p) for s in (1, -1) for p in legs)
    xml = f'''<mujoco model="ant">
<compiler angle="degree" meshdir="."/>
<option timestep="1e-4" gravity="0 0 -9810" cone="pyramidal" iterations="100" ls_iterations="50"/>
<default>
  <geom contype="0" conaffinity="0" density="1e-3" rgba="0.75 0.45 0.2 1"/>
  <joint damping="{DAMP}" armature="1e-4" limited="false"/>
</default>
<asset>
{chr(10).join(assets)}
</asset>
<worldbody>
<geom name="ground_plane" type="plane" size="200 200 1" rgba="0.85 0.75 0.55 1"/>
<body name="mesosoma" pos="0 0 {z:.3f}">
<freejoint name="root"/>
<geom name="mesosoma" type="mesh" mesh="mesosoma"/>
<body name="head" pos="2.2 0 0.25"><geom name="head" type="mesh" mesh="head"/></body>
<body name="petiole" pos="-1.8 0 0">
<joint name="petiole_pitch" axis="0 1 0"/>
<joint name="petiole_yaw" axis="0 0 1"/>
<geom name="petiole" type="mesh" mesh="petiole"/>
<body name="gaster" pos="-1.6 0 0.1"><geom name="gaster" type="mesh" mesh="gaster"/></body>
</body>
{legs}
</body>
</worldbody>
<contact>
{chr(10).join(f'<pair geom1="ground_plane" geom2="{g}" solref="2e-4 1" solimp="0.98 0.99 1e-5 0.5 3" friction="1 1" margin="0.001"/>' for g in pairs)}
</contact>
<actuator>
<position name="petiole_pitch" joint="petiole_pitch" kp="{KP}" forcelimited="true" forcerange="{-FMAX} {FMAX}"/>
<position name="petiole_yaw" joint="petiole_yaw" kp="{KP}" forcelimited="true" forcerange="{-FMAX} {FMAX}"/>
{chr(10).join(actuators)}
</actuator>
</mujoco>
'''
    (out / 'ant.xml').write_text(xml)
    return out / 'ant.xml'


if __name__ == '__main__':
    p = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    r = stand(p)
    print(p, r)
    sys.exit(0 if r['stands'] else 'FAIL ant: does not stand in MuJoCo')

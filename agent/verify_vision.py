"""Real eye render, scene intervention and full-graph CPU sensory gate."""
import json
import numpy as np
import mujoco
import imageio.v3 as iio
from agent.neural_single import EthologyArena
from agent.vision import EyeVision
from agent.neural_connectome import CPUBrain,ROOT

def verify():
    out=ROOT/'.work/neural-flies';arena=EthologyArena(seed=1)
    try:
        eyes0=arena.eyes();v=EyeVision();x0,_,info=v.step(eyes0)
        arena.sim.mj_data.qpos[3:7]=[0,0,0,1]
        mujoco.mj_forward(arena.sim.mj_model,arena.sim.mj_data)
        eyes1=arena.eyes();x1,threat1,_=v.step(eyes1)
        assert not np.array_equal(eyes0,eyes1)
        assert np.linalg.norm(x1-x0)>1e-4
        xb,tb,_=v.step(eyes1,blind=True);assert not xb.any() and not tb.any()
        brain=CPUBrain();inputs=np.zeros((2,8),np.float32);inputs[0,6:]=x0
        for _ in range(8):active=brain.step(inputs)
        assert active['counts'][0][1]>0 and active['counts'][1][1]==0
        brain.reset()
        for _ in range(8):null=brain.step(np.zeros((2,8),np.float32))
        assert null['counts']==[[0.,0.],[0.,0.]]
        iio.imwrite(out/'eye-camera-pair.png',np.concatenate(eyes0,axis=1))
        result={'status':'PASS','image_shape':list(eyes0.shape),'initial_drive':x0.tolist(),
                'rotated_drive':x1.tolist(),'image_difference_mean':float(np.abs(eyes0.astype(float)-eyes1).mean()),
                'vision_only_neural_counts':active['counts'],'blind_only_neural_counts':null['counts'],
                'scope':'real rendering to coarse visual ports, not retinal reconstruction or object recognition'}
        (out/'vision-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:arena.close()
if __name__=='__main__':verify()

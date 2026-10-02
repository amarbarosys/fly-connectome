"""Disclosed engineered motivations; no consciousness or biological-policy claim."""
import numpy as np
from collections import deque


class Desire:
    def __init__(self):
        self.progress=deque()
        self.turn_sign=1.
        self.turn_reason=None
        self.time_s=0.
        self.turn_until=-.2
        self.state={}

    def evaluate(self, hunger, inputs, position, mouth_contact, upright, dt=.01, *, wall_contact=False, wall_turn=1.):
        inputs=np.asarray(inputs,float)
        if inputs.shape!=(8,) or not np.isfinite(inputs).all():raise ValueError('invalid desire senses')
        hunger=float(np.clip(hunger,0,1));self.time_s+=dt
        pos=np.asarray(position,float)
        self.progress.append((self.time_s,pos[:2].copy()))
        while len(self.progress)>1 and self.progress[1][0]<=self.time_s-1.:
            self.progress.popleft()
        elapsed=self.time_s-self.progress[0][0]
        displacement=float(np.linalg.norm(pos[:2]-self.progress[0][1]))
        stalled=elapsed>=.99 and displacement<.3
        threat=float(max(inputs[4:6]));odor=float(max(inputs[:2]));taste=float(inputs[2])
        goal=('REST' if upright<.8 else 'AVOID_THREAT' if threat>.25 else
              'EAT' if mouth_contact and taste>.05 and hunger>.05 else
              'SEEK_FOOD' if hunger>.1 and odor>1e-5 else 'EXPLORE')
        strength={'REST':1.,'AVOID_THREAT':threat,'EAT':hunger,
                  'SEEK_FOOD':hunger,'EXPLORE':.35}[goal]
        walking=goal in ('EXPLORE','SEEK_FOOD')
        if walking and self.time_s>=self.turn_until+.2 and (wall_contact or stalled):
            self.turn_sign=float(np.sign(wall_turn) or 1.) if wall_contact else -self.turn_sign
            self.turn_reason='wall_contact' if wall_contact else 'low_horizontal_progress'
            self.turn_until=self.time_s+1.6
            self.progress.clear();self.progress.append((self.time_s,pos[:2].copy()))
        turning=self.time_s<self.turn_until and goal not in ('REST','EAT','AVOID_THREAT')
        self.state={'goal':goal,'strength':strength,'hunger':hunger,'threat':threat,
                    'food_cue':odor,'mouth_contact':bool(mouth_contact),'unstick_turn':turning,
                    'wall_contact':bool(wall_contact),'horizontal_progress_mm':displacement,
                    'turn_reason':self.turn_reason if turning else None,'turn_sign':self.turn_sign,
                    'turn_phase':('BACK_UP' if self.turn_until-self.time_s>1.2 else 'PIVOT') if turning else None,
                    'origin':'engineered motivational policy'}
        # Internal drives use existing input ports as a prototype, not as a
        # reconstruction of hunger/neuromodulator circuitry.
        driven=np.clip(inputs.copy(),0,1)
        if goal in ('EXPLORE','SEEK_FOOD'):
            driven[:2]=np.clip(driven[:2]*(1+2*hunger)+.35,0,1)
        return driven,self.state.copy()

    def modulate(self, output, signal):
        raw=np.asarray(output,float)
        if raw.shape!=(5,) or not np.isfinite(raw).all():raise ValueError('invalid desire decoder')
        if not signal:return np.zeros(5)
        out=raw.copy();goal=self.state['goal']
        if goal=='REST':return np.zeros(5)
        if goal=='EAT':out[:2]=0;out[2]=max(out[2],.55+.4*self.state['hunger']);out[3:]=0
        elif goal=='AVOID_THREAT':
            out[:2]=[.65,.15] if self.time_s%2<1 else [.15,.65]
            out[2:4]=0;out[4]=.8
        else:
            out[:2]=np.clip(out[:2],.25,.65);out[2:4]=0;out[4]=0
            if self.state['unstick_turn']:
                out[:2]=[-.5,-.5] if self.state['turn_phase']=='BACK_UP' else [-.5,.5] if self.turn_sign>0 else [.5,-.5]
        return np.clip(out,-.65,1.2)

"""Independently recompute batch costs and audit saved body observations."""
import json
from pathlib import Path
import numpy as np


def verify(root):
    rows = []; exchanges = 0
    for variant in ('baseline','optimized'):
        for count in (1,2,4,10):
            directory = root/f'{variant}-c{count}'
            metrics = json.loads((directory/'metrics.json').read_text())
            raw = json.loads((directory/'exchanges.json').read_text())
            assert metrics['status']=='PASS' and metrics['flies']==count
            assert len(raw)==metrics['steps']+metrics['warmup_steps']==23
            for index, row in enumerate(raw):
                assert row['step']==index and len(row['bodies'])==count
                assert row['loop_s'] >= row['neural_s']+row['decode_s']+row['physics_round_trip_s']-1e-8
                for lane, body in enumerate(row['bodies']):
                    assert body['index']==lane and body['physics_steps']==(index+1)*100
                    assert np.isfinite([*body['inputs'],*body['position'],body['upright'],body['ingested']]).all()
                    assert body['upright']>.85
                    exchanges+=1
            measured=raw[metrics['warmup_steps']:]
            seconds=sum(row['loop_s'] for row in measured)
            assert seconds==metrics['wall_seconds_measured']
            assert abs(count*metrics['steps']*.01/seconds-metrics['aggregate_fly_seconds_per_wall_second'])<1e-12
            for name in metrics['component_seconds']:
                assert abs(sum(row[name] for row in measured)-metrics['component_seconds'][name])<1e-12
            assert metrics['shared_decoder_updates']==0 and metrics['cpu_gpu_feature_error']<=1e-5
            rows.append({'variant':variant,'flies':count,'aggregate_rate':metrics['aggregate_fly_seconds_per_wall_second'],
                         'median_loop_ms':metrics['median_loop_ms'],'wall_seconds':seconds})
    for count in (1,2,4,10):
        a=json.loads((root/f'baseline-c{count}/exchanges.json').read_text())
        b=json.loads((root/f'optimized-c{count}/exchanges.json').read_text())
        assert all(x['bodies']==y['bodies'] for x,y in zip(a,b))
    directory=root/'full-ten-learning';m=json.loads((directory/'metrics.json').read_text())
    raw=json.loads((directory/'exchanges.json').read_text())
    assert len(raw)==138 and m['finite_body_exchanges']==1380 and m['status']=='PASS'
    for step,row in enumerate(raw):
        assert len(row['bodies'])==10
        for lane,body in enumerate(row['bodies']):
            assert body['index']==lane and body['physics_steps']==(step+1)*100
            assert np.isfinite([*body['inputs'],*body['position'],body['upright'],body['ingested']]).all()
            assert body['upright']>.85
            exchanges+=1
    assert all(value>0 for value in m['ingested_food_units_by_fly'])
    assert m['shared_decoder_updates']==sum(body['reward']>0 for row in raw for body in row['bodies'])
    before=np.load(root/'baseline-c10/readout.npz');after=np.load(directory/'readout.npz')
    change=float(np.linalg.norm(after['weights']-before['weights']))
    assert abs(change-m['readout_weight_delta_norm'])<1e-12 and change>0
    baseline=next(r for r in rows if r['variant']=='baseline' and r['flies']==10)
    optimized=next(r for r in rows if r['variant']=='optimized' and r['flies']==10)
    report={'status':'PASS','finite_body_exchanges_checked':exchanges,
        'before_after_observed_body_states_equal':True,'ten_fly_throughput_speedup':baseline['wall_seconds']/optimized['wall_seconds'],
        'full_ten_shared_updates':m['shared_decoder_updates'],'full_ten_food_intake':m['ingested_food_units_by_fly'],
        'cases':rows,'scope':'bounded odor-only headless capacity/feedback test, not training efficacy or biological validation'}
    (root/'independent-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':
    verify(Path('.work/agent/ten-fly'))

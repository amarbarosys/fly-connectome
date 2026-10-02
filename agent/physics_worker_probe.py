"""Compare CPU body concurrency outside the GPU queue using identical commands."""
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import numpy as np


def main():
    root = Path('.work/agent/ten-fly')
    for workers in (6, 8):
        socket = root/f'physics-probe-w{workers}.sock'
        env = dict(os.environ, MUJOCO_GL='egl', LIBGL_ALWAYS_SOFTWARE='1',
                   GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with (root/f'physics-probe-w{workers}-server.log').open('w') as log:
            process = subprocess.Popen([sys.executable,'-m','agent.batch_arenas','serve','--socket',str(socket),'--workers',str(workers)],
                                       env=env, stdout=subprocess.PIPE, stderr=log, text=True)
            connection = None
            try:
                selector = selectors.DefaultSelector(); selector.register(process.stdout,selectors.EVENT_READ)
                if not selector.select(60):
                    raise TimeoutError('CPU body server startup')
                ready = json.loads(process.stdout.readline()); assert ready['physics_ready']
                selector.close()
                connection = Client(str(socket),family='AF_UNIX')
                connection.send({'op':'reset','actors':[(i,None) for i in range(10)]});connection.recv()
                times=[]; last=None
                for step in range(30):
                    commands=[(i,('IDLE',[0.,0.]) if step<8 else ('WALK',[.2,.65] if i%2 else [.65,.2])) for i in range(10)]
                    started=time.perf_counter();connection.send({'op':'step','actors':commands});last=connection.recv();times.append(time.perf_counter()-started)
                report={'workers':workers,'median_physics_ms':float(np.median(times[3:])*1000),
                    'p95_physics_ms':float(np.quantile(times[3:],.95)*1000),'total_physics_s':sum(times[3:]),
                    'final_bodies':last['rows'],'worker_rss':last['body_worker_rss']}
                (root/f'physics-probe-w{workers}.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:v for k,v in report.items() if k!='final_bodies'}),flush=True)
            finally:
                if connection is not None:
                    connection.send({'op':'shutdown'});connection.recv();connection.close()
                process.wait(timeout=20)
    candidates=[json.loads((root/f'physics-probe-w{w}.json').read_text()) for w in (4,6,8)]
    for row in candidates[1:]:
        assert row['final_bodies']==candidates[0]['final_bodies'], 'concurrency changed body trajectory'
    winner=min(candidates,key=lambda r:r['total_physics_s'])['workers']
    (root/'physics-worker-choice.json').write_text(json.dumps({'winner':winner,'final_state_equal':True,
        'candidates':[{k:v for k,v in r.items() if k!='final_bodies'} for r in candidates]},indent=2)+'\n')


if __name__=='__main__':
    main()

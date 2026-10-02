"""One queued sweep of the preflighted independent-arena batch sizes."""
import argparse
import json
from pathlib import Path
import numpy as np
from agent.batch_arenas import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('baseline', 'optimized'), required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root = Path('.work/agent/ten-fly')
    for count in (1, 2, 4, 10):
        assert (root/f'bin/worker-c{count}-{args.variant}').is_file()
        arrays = np.load(root/f'fixture/oracle-c{count}.npz')
        assert arrays['inputs'].shape == (3, count, 8)
        assert arrays['features'].shape == (3, count, 2606)
        assert np.isfinite(arrays['features']).all()
    if args.preflight:
        print(json.dumps({'status': 'PASS', 'variant': args.variant, 'counts': [1,2,4,10]}))
        return
    for count in (1, 2, 4, 10):
        benchmark(root/'physics.sock', root/'fixture', root/f'{args.variant}-c{count}',
                  count, root/f'bin/worker-c{count}-{args.variant}', steps=20, warmup=3)


if __name__ == '__main__':
    main()

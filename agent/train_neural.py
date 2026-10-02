"""One bounded GPU job: parity/isolation gate, then readout training."""
from agent.verify_neural import verify
from agent.neural_connectome import train

if __name__ == '__main__':
    verify()
    report = train('gpu', pairs=80)
    if report['rmse'] >= report['constant_rmse']:
        raise SystemExit('Readout failed held-out constant-baseline gate; do not deploy it.')

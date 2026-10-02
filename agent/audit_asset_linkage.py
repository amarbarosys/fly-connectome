"""Read-only asset linkage audit: which raw layers are consumed by code, not just present."""
import json
from pathlib import Path
import subprocess
import time
from agent.neural_connectome import ROOT,digest

ASSETS={'connectome_csr':('data/row_ptr.bin','row_ptr'),
    'connectome_edges':('data/proofread_connections_783.feather','proofread_connections_783'),
    'l1_synapse_coords':('data/flywire_synapses_783.feather','flywire_synapses_783'),
    'l2_skeletons':('data/sk_lod1_783_healed_ds2.parquet','sk_lod1_783'),
    'l4_cell_atlas':('data/fca_head.h5ad','fca_head'),
    'l5_calcium_nwb':('data/sub-13_ses-20201201_behavior_ophys.nwb','sub-13_ses')}
EXTRACTS=['synapse-prefix.npz','skeleton-prefix.npz','fca-expression-reference.npz','calcium-behavior-reference.npz']


SELF=f'agent/{Path(__file__).name}'
WRITER='agent/fly_asset_activity.py'


def consumers(token):
    """Files referencing token. Excludes this auditor: it names every token it greps for,
    so counting itself would report every asset as linked."""
    done=subprocess.run(['grep','-rl',token,'--include=*.py','--include=*.mojo','--include=*.rs','.'],
        cwd=ROOT,capture_output=True,text=True)
    assert done.returncode in (0,1),f'grep {token} exit {done.returncode}'
    return sorted(p[2:] for p in done.stdout.split() if not p.startswith('./.work') and p[2:]!=SELF)


def audit():
    assets={}
    for name,(path,token) in ASSETS.items():
        source=ROOT/path;found=consumers(token)
        assets[name]={'path':path,'present':source.exists(),
            'bytes':source.stat().st_size if source.exists() else 0,
            'consumers':found,'consumer_count':len(found),'linked':len(found)>1}
    # a writer reading its own output back for a manifest is not a downstream consumer
    extracts={}
    for name in EXTRACTS:
        readers=consumers(name);downstream=[p for p in readers if p!=WRITER]
        extracts[name]={'readers':readers,'downstream_readers':downstream,'dead_end':not downstream}
    report={'status':'PASS','assets':assets,'extracts':extracts,
        'linked_count':sum(row['linked'] for row in assets.values()),'asset_count':len(assets),
        'scope':'static source-reference audit over tracked py/mojo/rs files; a reference is not proof of use at runtime, '
            'and absence of a reference is proof only for these three extensions outside .work',
        'crosswalk_established':False,
        'sources_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
            [ROOT/'agent/fly_asset_activity.py',ROOT/'agent/calcium_validation.py',Path(__file__)]},
        'recorded_unix':time.time()}
    out=ROOT/'.work/agent/asset-linkage/linkage.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':print(json.dumps(audit()))

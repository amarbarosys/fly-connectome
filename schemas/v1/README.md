# Fly screen contracts, version 1

- `compound-record.schema.json` records source-facing target and assay evidence. `Ki`, `Kd`, `IC50`, and `EC50` remain distinct and require explicit concentration units and assay context.
- `perturbation-profile.schema.json` matches the current simulator boundary: global gains scale already-signed presynaptic transmitter edges. It does not claim receptor-level or neuron-specific action.
- `screening-manifest.schema.json` captures conditions, engine identity, complete input hashes, dose basis, and requested steps. A run receipt will add measured outputs and execution/host metadata.

Schema validation checks shape only. It cannot establish source credibility, assay comparability, target mapping, biological validity, or that the engine applied the stated semantics. The version-1 profile shape mirrors the in-progress `pharma_contracts.py` contract; coordinate changes with its owner before either becomes a released interface.

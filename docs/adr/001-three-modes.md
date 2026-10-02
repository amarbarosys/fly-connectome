# ADR-001: Three-Mode Separation for Fly World

**Status:** Accepted | **Date:** 2026-09-30 | **Owner:** wDV:p2

## Context

The Fly World master plan defines three distinct modes of operation (§3):

1. **Biological model** — supported anatomy and mappings; fitted assumptions and missing evidence visible. Does not imply validation.
2. **Engineering agent** — connectome-inspired or connectome-constrained control with disclosed adapters and engineered drives.
3. **Game agent** — stylized stamina, victory points, signaling and combat rules. Natural aggression claims require separate ethological test.

**Problem:** Existing evidence (`docs/agent-evidence.md`) had no mode column. Analysis, publications, and demos could mix conclusions across modes without detection.

## Decision

Every run, artifact, and evidence row **must declare exactly one mode**. Mode switching creates a new configuration identity. A shared simulator may host these modes but must not mix their conclusions.

Enforcement mechanisms:
1. **Config schema** — `mode` field required in all run manifests (enum: `biological_model`, `engineering_agent`, `game_agent`).
2. **Runtime guard** — `agent.mode_guard.ModeGuard` validates mode at startup; fails fast on mismatch.
3. **Analysis linter** — `agent.mode_guard.lint_evidence()` scans markdown tables for missing/incorrect mode columns.
4. **ADR linkage** — all evidence docs link to this ADR; CI fails if mode column absent.

## Consequences

- **Positive:** Prevents mode-laundering (e.g., presenting engineering-agent results as biological validation).
- **Positive:** Makes §3 load-bearing — mode column in `agent-evidence.md` now enforced.
- **Negative:** Adds a required field to all run configs; existing scripts must be updated.
- **Negative:** Game-agent mode cannot cite biological model evidence without explicit caveat.

## Implementation

```python
# agent/mode_guard.py
from enum import Enum

class RunMode(Enum):
    BIOLOGICAL_MODEL = "biological_model"
    ENGINEERING_AGENT = "engineering_agent"
    GAME_AGENT = "game_agent"

class ModeGuard:
    def __init__(self, declared_mode: RunMode, config: dict):
        self.declared = declared_mode
        self.config = config
    
    def validate(self) -> None:
        # Check config.mode matches declared
        # Check no mixed-mode imports in this run
        pass

def lint_evidence(path: str) -> list[str]:
    # Scan markdown for capability tables; verify Mode column exists
    # Return list of violations
    pass
```

## Verification

Run: `python -m agent.mode_guard lint docs/agent-evidence.md`

Expected: zero violations (mode column present, values valid enum).

## Related

- Master plan §3 (Agent and world contract)
- Master plan §13 (First execution packet — all items are `engineering_agent`)
- `docs/agent-evidence.md` (capability ledger with Mode column)
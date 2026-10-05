"""Standalone deterministic gate for the game state module and event replay."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from agent.body_engine.game_events import _self_check


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "agent/body_engine/game_state.mojo"


def main() -> None:
    mojo = Path.home() / ".local/bin/mojo"
    with tempfile.TemporaryDirectory(prefix="game-state-check-") as tmp:
        binary = Path(tmp) / "game_state"
        subprocess.run([str(mojo), "build", str(MODULE), "-o", str(binary)],
                       cwd=ROOT, check=True)
        subprocess.run([str(binary)], cwd=ROOT, check=True)
    _self_check()


if __name__ == "__main__":
    main()

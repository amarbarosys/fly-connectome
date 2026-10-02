"""Three-mode enforcement for Fly World.

ADR-001: Every run declares exactly one mode. Mode switching = new config identity.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any


class RunMode(Enum):
    BIOLOGICAL_MODEL = "biological_model"
    ENGINEERING_AGENT = "engineering_agent"
    GAME_AGENT = "game_agent"

    @classmethod
    def from_str(cls, s: str) -> RunMode:
        try:
            return cls(s.lower().replace(" ", "_"))
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Invalid mode '{s}'. Valid: {valid}")

    def __str__(self) -> str:
        return self.value


class ModeGuard:
    """Validates run configuration against declared mode."""

    def __init__(self, declared_mode: RunMode, config: dict[str, Any]):
        self.declared = declared_mode
        self.config = config

    def validate(self) -> None:
        """Fail fast if config contradicts declared mode."""
        config_mode = self.config.get("mode")
        if config_mode is not None:
            config_mode = RunMode.from_str(config_mode)
            if config_mode != self.declared:
                raise ValueError(
                    f"Config mode '{config_mode}' != declared mode '{self.declared}'"
                )

        # Mode-specific forbidden keys (prevent accidental mixing)
        forbidden = {
            RunMode.BIOLOGICAL_MODEL: ["victory_points", "stamina", "strike_cooldown"],
            RunMode.ENGINEERING_AGENT: ["subjective_desire", "natural_aggression"],
            RunMode.GAME_AGENT: ["biological_validation", "wetlab_endpoint"],
        }
        for key in forbidden.get(self.declared, []):
            if key in self.config:
                raise ValueError(
                    f"Mode {self.declared} forbids config key '{key}' (mode mixing)"
                )

    @staticmethod
    def from_manifest(manifest_path: Path) -> ModeGuard:
        """Load mode from manifest.json and create guard."""
        import json
        data = json.loads(manifest_path.read_text())
        mode = RunMode.from_str(data["mode"])
        return ModeGuard(mode, data)


# Markdown table linter
_CAPABILITY_TABLE_RE = re.compile(
    r"^\| Capability \| Mode \| Supported evidence \| Open claim \|",
    re.MULTILINE,
)


def lint_evidence(path: str | Path) -> list[str]:
    """Scan evidence markdown for capability tables; verify Mode column exists and values valid.

    Returns list of violation strings (empty = clean).
    """
    text = Path(path).read_text()
    violations = []

    # Find capability ledger tables
    for match in _CAPABILITY_TABLE_RE.finditer(text):
        table_start = match.start()
        # Find table end (next blank line or heading)
        table_end = text.find("\n\n", table_start)
        if table_end == -1:
            table_end = len(text)
        table_text = text[table_start:table_end]

        # Parse rows
        lines = [l.strip() for l in table_text.split("\n") if l.strip().startswith("|")]
        if len(lines) < 3:
            violations.append(f"{path}: capability table too short")
            continue

        header = [c.strip() for c in lines[0].split("|")[1:-1]]
        if "Mode" not in header:
            violations.append(f"{path}: capability table missing 'Mode' column")
            continue
        mode_idx = header.index("Mode")

        for i, line in enumerate(lines[2:], 1):  # skip header + separator
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) <= mode_idx:
                violations.append(f"{path}: row {i} missing mode cell")
                continue
            mode_val = cells[mode_idx]
            try:
                RunMode.from_str(mode_val)
            except ValueError as e:
                violations.append(f"{path}: row {i} invalid mode '{mode_val}': {e}")

    return violations


def lint_all_evidence(root: str | Path = ".") -> dict[str, list[str]]:
    """Lint all evidence markdown files under root."""
    root = Path(root)
    results = {}
    for md_file in root.rglob("*evidence*.md"):
        violations = lint_evidence(md_file)
        if violations:
            results[str(md_file)] = violations
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "lint":
        target = sys.argv[2] if len(sys.argv) > 2 else "docs/agent-evidence.md"
        violations = lint_evidence(target)
        if violations:
            for v in violations:
                print(f"VIOLATION: {v}")
            sys.exit(1)
        else:
            print(f"OK: {target} passes mode lint")
            sys.exit(0)
    else:
        print("Usage: python -m agent.mode_guard lint [path]")
        sys.exit(2)
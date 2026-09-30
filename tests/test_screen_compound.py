import contextlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

import screen_compound


class ScreenCompoundTests(unittest.TestCase):
    def test_missing_and_invalid_telemetry_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "missing required metric"):
            screen_compound.require_telemetry({"steps": 2})
        with self.assertRaisesRegex(RuntimeError, "requested 3"):
            screen_compound.require_telemetry({
                "steps": 2, "step_ms": 1.0, "mean_active": 3,
                "mean_dn": 1, "synchrony": 0.2,
            }, requested_steps=3)

    def test_zero_control_metric_has_no_percentage_fallback(self):
        with self.assertRaisesRegex(ValueError, "neutral-control synchrony is zero"):
            screen_compound.percent_delta(0, 1, "synchrony")

    def test_scorecard_uses_calibrated_language_and_defines_synchrony(self):
        base = {"steps": 2, "step_ms": 1.0, "mean_active": 10,
                "mean_dn": 2, "synchrony": 0.25}
        test = {**base, "compound": "Fixture", "mean_active": 12,
                "mean_dn": 1, "synchrony": 0.5,
                "delta_active_pct": 20, "delta_dn_pct": -50,
                "delta_synchrony_pct": 100,
                "interpretation": "modeled activity increased"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            screen_compound.print_scorecard(test, base)
        report = out.getvalue()
        self.assertIn("not a biological phenotype claim", report)
        self.assertIn("population-count SD / (mean + 1)", report)
        for unsupported in ("EPILEPTIFORM", "PARALYTIC", "TOLERATED"):
            self.assertNotIn(unsupported, report)

    def test_cli_stub_end_to_end_honors_steps_for_control_and_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            capture = root / "calls.txt"
            profile = root / "fixture.json"
            profile.write_text(json.dumps({"name": "Fixture", "class": "hypothesis", "gains": {"ach": 1.2}}))

            gpu_wait = bin_dir / "gpu-wait"
            gpu_wait.write_text("#!/bin/sh\nwhile [ \"$1\" != \"--\" ]; do shift; done\nshift\nexec \"$@\"\n")
            engine = bin_dir / "fake-pharma-engine"
            engine.write_text("""#!/bin/sh
printf '%s:%s\\n' "$PHARMA_COMPOUND" "$PHARMA_STEPS" >> "$CAPTURE_PATH"
printf 'PHARMA_RESULT_COMPOUND=%s\\n' "$PHARMA_COMPOUND"
printf 'PHARMA_RESULT_STEPS=%s\\n' "$PHARMA_STEPS"
printf 'PHARMA_RESULT_STEP_MS=2.5\\n'
if [ "$PHARMA_COMPOUND" = "Baseline" ]; then
  printf 'PHARMA_RESULT_MEAN_ACTIVE=10\\nPHARMA_RESULT_MEAN_DN=2\\nPHARMA_RESULT_SYNCHRONY=0.5\\n'
else
  printf 'PHARMA_RESULT_MEAN_ACTIVE=12\\nPHARMA_RESULT_MEAN_DN=1\\nPHARMA_RESULT_SYNCHRONY=0.75\\n'
fi
""")
            gpu_wait.chmod(gpu_wait.stat().st_mode | stat.S_IXUSR)
            engine.chmod(engine.stat().st_mode | stat.S_IXUSR)

            out = io.StringIO()
            with patch.object(screen_compound, "PHARMA_BIN_DIR", str(root)), \
                 patch.object(screen_compound, "PHARMA_BIN", str(engine)), \
                 patch.dict(os.environ, {"PATH": f"{bin_dir}:{os.environ['PATH']}", "CAPTURE_PATH": str(capture)}), \
                 patch.object(sys, "argv", ["screen_compound.py", "--steps", "7", str(profile)]), \
                 contextlib.redirect_stdout(out):
                screen_compound.main()

            self.assertEqual(capture.read_text().splitlines(), ["Baseline:7", "Fixture:7"])
            self.assertIn("steps=7", out.getvalue())
            self.assertIn("Fixture", out.getvalue())
            self.assertIn("All compounds screened successfully.", out.getvalue())


if __name__ == "__main__":
    unittest.main()

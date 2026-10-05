"""CPU compile and framing gate for follow_camera.mojo, including target-offset mutant."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
MOJO = Path.home() / ".local/bin/mojo"
SOURCE = ROOT / "follow_camera.mojo"
HUD = ROOT / "hud.mojo"


def main():
    with tempfile.TemporaryDirectory(prefix="spectator-camera-") as td:
        binary = Path(td) / "follow_camera"
        built = subprocess.run([str(MOJO), "build", str(SOURCE), "-o", str(binary)], text=True, capture_output=True)
        if built.returncode:
            raise SystemExit(built.stderr)
        good = subprocess.run([str(binary)], text=True, capture_output=True)
        if good.returncode or good.stdout.count("PASS camera fixture") != 2:
            raise SystemExit(f"FAIL camera fixture: {good.stdout}{good.stderr}")
        print(good.stdout, end="")

        original = SOURCE.read_text()
        needle = "tx /= Float32(n)"
        if original.count(needle) != 1:
            raise SystemExit("FAIL mutant setup: expected one target-centering expression")
        mutant_source = Path(td) / "mutant.mojo"
        mutant_source.write_text(original.replace(needle, "tx = tx / Float32(n) + 1.0"))
        mutant_binary = Path(td) / "mutant"
        mutant_build = subprocess.run([str(MOJO), "build", str(mutant_source), "-o", str(mutant_binary)],
                                      text=True, capture_output=True)
        if mutant_build.returncode:
            raise SystemExit(f"FAIL mutant compile: {mutant_build.stderr}")
        mutant = subprocess.run([str(mutant_binary)], text=True, capture_output=True)
        diagnostic = mutant.stdout + mutant.stderr
        if mutant.returncode == 0 or "camera target missed" not in diagnostic:
            raise SystemExit(f"FAIL off-center target mutant was not rejected: {diagnostic}")
        print("PASS off-center target mutant rejected")

        hud_binary = Path(td) / "hud"
        hud_build = subprocess.run([str(MOJO), "build", str(HUD), "-o", str(hud_binary)],
                                   text=True, capture_output=True)
        if hud_build.returncode:
            raise SystemExit(hud_build.stderr)
        hud = subprocess.run([str(hud_binary)], text=True, capture_output=True)
        if hud.returncode or "PASS HUD bars: energy 90 px, health 30 px, stamina 60 px" not in hud.stdout:
            raise SystemExit(f"FAIL HUD bar fixture: {hud.stdout}{hud.stderr}")
        if "PASS source marker: descriptor position and kind preserved" not in hud.stdout:
            raise SystemExit(f"FAIL source marker fixture: {hud.stdout}{hud.stderr}")
        print(hud.stdout, end="")


if __name__ == "__main__":
    main()

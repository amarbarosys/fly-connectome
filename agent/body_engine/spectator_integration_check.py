"""Receipt-driven integration gates for the camera/HUD live hook.

Usage after live/render wiring has produced its artifacts:
  python spectator_integration_check.py receipt.json

Receipt paths are relative to the JSON file. See README.md in the handoff directory.
"""
import json
import sys
from pathlib import Path

import numpy as np


def path_for(receipt_path, value):
    path = Path(value)
    return path if path.is_absolute() else receipt_path / path


def require(ok, message):
    if not ok:
        raise SystemExit(f"FAIL {message}")
    print(f"PASS {message}")


def baseline_gate(receipt, base):
    reference = path_for(base, receipt["baseline_reference_rgba"]).read_bytes()
    disabled = path_for(base, receipt["baseline_disabled_rgba"]).read_bytes()
    require(bool(reference) and reference == disabled,
            f"baseline frame unchanged with spectator hooks disabled ({len(reference)} bytes, byte-equal)")


def moving_clip_gate(receipt, base):
    w, h, n = int(receipt["width"]), int(receipt["height"]), int(receipt["clip_frames"])
    image_path = path_for(base, receipt["moving_clip_rgba"])
    image = np.fromfile(image_path, dtype=np.uint8)
    require(image.size == n * w * h * 4, f"moving clip has {n} complete {w}x{h} RGBA frames")
    image = image.reshape(n, h, w, 4)
    root_path = path_for(base, receipt["root_trace_f32"])
    roots = np.fromfile(root_path, dtype=np.float32)
    animals = int(receipt["animals"])
    require(roots.size == n * animals * 3, f"root trace has {n} frames for {animals} animals")
    roots = roots.reshape(n, animals, 3)
    travel = np.linalg.norm(roots[-1] - roots[0], axis=1)
    min_travel = float(receipt.get("min_root_travel_mm", 1.0))
    require(float(travel.max()) >= min_travel,
            f"moving clip contains root travel {float(travel.max()):.3f} mm (minimum {min_travel:.3f})")
    centers = []
    for i, frame in enumerate(image):
        ys, xs = np.nonzero(frame[..., 3] == 255)
        if not len(xs):
            raise SystemExit(f"FAIL moving clip frame {i} contains no rendered animal pixels")
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        if x0 < 2 or y0 < 2 or x1 >= w - 2 or y1 >= h - 2:
            raise SystemExit(f"FAIL moving clip frame {i} clips animal bbox {(x0, y0, x1, y1)}")
        cx, cy = float(xs.mean() / w), float(ys.mean() / h)
        if not 0.15 <= cx <= 0.85 or not 0.15 <= cy <= 0.85:
            raise SystemExit(f"FAIL follow framing frame {i} centers animal at {(cx, cy)}")
        centers.append((cx, cy))
    print(f"PASS follow camera frames all {n} poses without clipping; roots travel {travel.tolist()} mm")


def cost_gate(receipt):
    costs = receipt["frame_cost_ms"]
    required = ("physics", "neural_control", "render", "copy", "display")
    missing = [name for name in required if name not in costs]
    require(not missing, f"frame-cost receipt reports separate phases: {', '.join(required)}")
    values = [float(costs[name]) for name in required]
    require(all(np.isfinite(values)) and all(x >= 0 for x in values),
            "frame-cost phase values are finite and nonnegative")
    fps, rtf = float(receipt["achieved_fps"]), float(receipt["real_time_factor"])
    require(np.isfinite(fps) and fps > 0 and np.isfinite(rtf) and rtf > 0,
            f"achieved fps {fps:.3f}, real-time factor {rtf:.6g}")
    semantics = receipt["animal_count_semantics"]
    require(semantics in ("batch instances", "bodies in a shared physical scene"),
            f"animal count semantics declared: {semantics}")
    count = int(receipt["rendered_animals"])
    require(count in (1, 8, 64), f"frame-cost receipt covers {count} rendered animals")
    print("FRAME COST ms/frame " + " ".join(f"{name}={float(costs[name]):.4f}" for name in required))


def leak_gate(receipt):
    leak = receipt["leak"]
    early, late = int(leak["sample_frames"][0]), int(leak["sample_frames"][1])
    require(early == 100 and late == 10_000, "leak samples taken at frames 100 and 10,000")
    fds = leak["fds"]
    vram = leak["vram_kib"]
    rss = leak["rss_kib"]
    ok = len(fds) == len(vram) == len(rss) == 2 and fds[0] == fds[1] and vram[1] <= vram[0] and rss[1] - rss[0] < 1024
    require(ok, f"bounded leak: fds {fds}, VRAM KiB {vram}, RSS KiB {rss}")


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: spectator_integration_check.py RECEIPT.json")
    receipt_path = Path(argv[1]).resolve()
    receipt = json.loads(receipt_path.read_text())
    baseline_gate(receipt, receipt_path.parent)
    moving_clip_gate(receipt, receipt_path.parent)
    cost_gate(receipt)
    leak_gate(receipt)
    print("PASS spectator integration receipt: baseline, moving clip, frame cost, leak")


if __name__ == "__main__":
    main(sys.argv)

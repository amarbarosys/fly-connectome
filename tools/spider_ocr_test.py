"""spider-ocr: a jumping-spider gaze front end for fly-ocr (jerryjliu/fly_ocr, in .work/fly-ocr).

fly-ocr reads glyphs it is handed; its README says the fly "does not decide where to look". This
adds the looking, after the secondary/principal split in Winsor et al. 2023:
  periphery  whole page at 1/8 scale, ink energy, merged into line blobs (secondary eyes)
  saccades   blobs visited strongest first (winner-take-all order, as in spider_vision_test.py)
  fovea      full-resolution crop of each blob only, read by fly-ocr's letter model (principal eyes)

Test page: 4 invented lines in Fira Sans (a font fly-ocr held out of training) on a blank-heavy
page with ink specks. Score = character error rate (CER) vs truth and fraction of page pixels read.
S1: spider CER within 5 points of fly-ocr on the full page, reading <= 25% of the pixels.
S2: random fovea boxes of the same sizes (3 draws) must read worse than the spider.
For fun, not science: one synthetic page, one font, gaze rules hand-written.
First run: 15 saccades, 11 of them on specks (the width filter ran after dilation widened each
speck), each read as "W", CER 45%. The filter now counts raw ink cells before dilation.

Run with fly-ocr's venv: .work/fly-ocr/.venv/bin/python tools/spider_ocr_test.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FLY = Path(__file__).resolve().parents[1] / ".work/fly-ocr"
sys.path.insert(0, str(FLY / "src"))
from flyocr.pdf.text import prepare_text  # noqa: E402
from flyocr.readout.predict import Recognizer  # noqa: E402

LINES = [((140, 120), "Spiders read pages"), ((820, 330), "Jumping toward text"),
         ((260, 640), "Eight eyes one fovea"), ((900, 900), "Small brain big view")]
PAGE, SCALE, PAD = (1600, 1100), 8, 12


def make_page():
    img = Image.new("L", PAGE, 255)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FLY / "assets/fonts/firasans/FiraSans-Regular.ttf"), 44)
    for xy, text in LINES:
        d.text(xy, text, font=font, fill=20)
    rng = np.random.default_rng(3)
    for x, y in rng.integers((0, 0), PAGE, (12, 2)):  # specks the periphery should ignore
        d.ellipse((x, y, x + 3, y + 3), fill=60)
    return img


def periphery(page):
    a = 255 - np.asarray(page, np.float32)
    h, w = a.shape[0] // SCALE, a.shape[1] // SCALE
    small = a[:h * SCALE, :w * SCALE].reshape(h, SCALE, w, SCALE).mean((1, 3))
    ink_cells = small > 12
    hot = ndimage.binary_dilation(ink_cells, np.ones((1, 5)))  # merge letters into line blobs
    lab, _ = ndimage.label(hot)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if ink_cells[sl][lab[sl] == i].sum() < 6:  # too little raw ink to be text: a speck
            continue
        ink = small[sl][lab[sl] == i].sum()
        y0, y1 = sl[0].start * SCALE - PAD, sl[0].stop * SCALE + PAD
        x0, x1 = sl[1].start * SCALE - PAD, sl[1].stop * SCALE + PAD
        boxes.append((ink, (max(x0, 0), max(y0, 0), min(x1, PAGE[0]), min(y1, PAGE[1]))))
    return [b for _, b in sorted(boxes, reverse=True)]  # saccade order: strongest first


def read(rec, img, tmp):
    out = Path(tempfile.mkdtemp(dir=tmp))
    img.save(out / "in.png")
    try:
        seg = prepare_text(str(out / "in.png"), out)
    except Exception as e:  # fly-ocr segmentation rejects regions with no text
        return [], 0, f"{type(e).__name__}: {e}"
    lines, n = [], 0
    for row in seg["rows"]:
        text = "".join(g.get("prefix", "") + rec.recognize(np.asarray(Image.open(out / g["image"]).convert("L")))["character"]
                       for g in row["glyphs"])
        n += len(row["glyphs"])
        lines.append((row["box"][1], text))
    return lines, n, None


def cer(lines, truth):
    a = " ".join(" ".join(t for _, t in sorted(lines)).split())
    b = " ".join(truth)
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d.copy(), i
        for j, cb in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (ca != cb))
    return d[-1] / len(b)


def run_boxes(rec, page, boxes, tmp):
    lines, glyphs = [], 0
    for box in boxes:
        got, n, _ = read(rec, page.crop(box), tmp)
        lines += [(box[1] + y, t) for y, t in got]
        glyphs += n
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes) + PAGE[0] * PAGE[1] / SCALE ** 2
    return lines, glyphs, area / (PAGE[0] * PAGE[1])


def main():
    rec = Recognizer(str(FLY / "data/graph"), str(FLY / "artifacts/letters-v2"))
    page, truth = make_page(), [t for _, t in sorted(LINES, key=lambda l: l[0][1])]
    tmp = FLY / ".cache/spider-ocr"
    tmp.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    full, n_full, err = read(rec, page, tmp)
    c_full = cer(full, truth)
    print(f"fly-ocr full page: CER {c_full:.1%}, {n_full} glyphs, 100% pixels, "
          f"{time.perf_counter() - t:.0f} s{', segmentation failed: ' + err if err else ''}")
    for _, line in sorted(full):
        print(f"   {line}")

    t = time.perf_counter()
    boxes = periphery(page)
    spider, n_sp, frac = run_boxes(rec, page, boxes, tmp)
    c_sp = cer(spider, truth)
    print(f"spider-ocr: {len(boxes)} saccades, CER {c_sp:.1%}, {n_sp} glyphs, {frac:.1%} pixels, "
          f"{time.perf_counter() - t:.0f} s")
    for _, line in sorted(spider):
        print(f"   {line}")
    ok1 = c_sp <= c_full + 0.05 and frac <= 0.25
    print(f"S1 accuracy vs full page and pixel budget -> {'PASS' if ok1 else 'FAIL'}")

    rng, c_rand = np.random.default_rng(4), []
    for _ in range(3):
        rand = []
        for x0, y0, x1, y1 in boxes:
            w, h = x1 - x0, y1 - y0
            x, y = int(rng.integers(0, PAGE[0] - w)), int(rng.integers(0, PAGE[1] - h))
            rand.append((x, y, x + w, y + h))
        c_rand.append(cer(run_boxes(rec, page, rand, tmp)[0], truth))
    ok2 = min(c_rand) > c_sp
    print(f"S2 random fovea boxes: CER {', '.join(f'{c:.1%}' for c in c_rand)} vs spider {c_sp:.1%} "
          f"-> {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        print(f"FAIL {2 - ok1 - ok2}/2 checks")
        sys.exit(1)


if __name__ == "__main__":
    main()

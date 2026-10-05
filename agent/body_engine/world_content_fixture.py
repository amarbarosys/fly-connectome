"""Write the worldsense review image and concentration trace under .work."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agent.body_engine import world_content as wc


def main():
    out = Path(".work/agent/body_engine/world-content")
    out.mkdir(parents=True, exist_ok=True)
    world = wc.build(0)
    x = np.linspace(-100, 1800, 180)
    y = np.linspace(-900, 900, 180)
    xx, yy = np.meshgrid(x, y)
    c = wc.concentration_xy_mm(np.stack((xx.ravel(), yy.ravel()), -1), world).reshape(*xx.shape, 2)
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax.set_title("Seeded plume sources (world mm)")
    ax.set_xlim(-100, 1800); ax.set_ylim(-900, 900); ax.set_aspect("equal")
    for src in world.sources:
        ax.scatter(src.x_mm, src.y_mm, s=110, color=np.asarray(src.color_rgb) / 255,
                   edgecolor="black", linewidth=0.7)
        ax.text(src.x_mm + 45, src.y_mm + 30, "food" if src.kind == wc.FOOD else "decoy")
    ax.arrow(1000, 760, 350, 0, width=4, head_width=45, head_length=55, color="#444")
    ax.text(1000, 810, "wind +x")
    heat.set_title("Food concentration minus decoy concentration")
    heat.imshow(c[..., 0] - c[..., 1], origin="lower", extent=(x.min(), x.max(), y.min(), y.max()),
                cmap="coolwarm", vmin=-0.8, vmax=0.8, aspect="auto")
    heat.set_xlabel("x (mm)"); heat.set_ylabel("y (mm)")
    fig.savefig(out / "plume-layout-and-field.png", dpi=160)
    poses = np.asarray([[300, 340], [600, 340], [900, 340], [1200, 340],
                        [300, -360], [600, -360], [900, -360], [1200, -360]], np.float32)
    vals = wc.concentration_xy_mm(poses, world)
    np.savetxt(out / "concentration-trace.tsv", np.c_[poses, vals], delimiter="\t",
               header="x_mm\ty_mm\tfood\tdecoy", comments="")
    print(f"PASS wrote {out}/plume-layout-and-field.png and concentration-trace.tsv")


if __name__ == "__main__":
    main()

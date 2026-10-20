"""Draw the shared-contract training and paired-validation schematic."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "paper" / "figures" if HERE.name == "scripts" else HERE
QA = ROOT / "qa"
QA.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "fidelityloop-framework",
})
W, H = 237.6, 154.8
fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
ax.axis("off")
ink, blue, green = "#263444", "#40749B", "#4D8077"
labels = []


def text(x, y, value, size=7, weight="normal", **kwargs):
    t = ax.text(x, y, value, ha="center", va="center", fontsize=size,
                fontweight=weight, color=ink, linespacing=1.15, **kwargs)
    labels.append(t)
    return t


def box(x, y, w, h, value, edge, fill, size=7):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
                          linewidth=.65, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    t = text(x + w / 2, y + h / 2, value, size)
    return patch, t


def arrow(start, end, color=ink, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6,
                                linewidth=.65, color=color,
                                shrinkA=0, shrinkB=0, **kwargs))


containers = []
containers.append(box(8, 137, 96, 14, "CPU replay training", blue, "#F1F6FA"))
containers.append(box(125, 137, 105, 14, "Frozen PPO checkpoint", blue, "#F1F6FA"))
arrow((106, 144), (123, 144), blue)
arrow((177.5, 136), (177.5, 129), blue)
text(W / 2, 123, "Rules or frozen PPO; shared policy contract", 6.8)

for center, color, fill, role, backend in [
    (60, blue, "#F4F8FB", "replay", "Replay backend\nService predictor"),
    (178, green, "#F4F9F7", "physical", "Execution backend\nvLLM / SGLang"),
]:
    arrow((center, 116), (center, 105), color)
    containers.append(box(center - 42, 83, 84, 22,
                          f"Policy adapter\n({role})", color, fill))
    arrow((center, 82), (center, 69), color)
    text(center + 18, 75.5, "action", 6.5)
    containers.append(box(center - 42, 45, 84, 24, backend, color, fill))
    # Each return path carries state from that loop's own backend only.
    outer = center - 51 if role == "replay" else center + 51
    near = center - 43 if role == "replay" else center + 43
    ax.plot([near, outer, outer], [57, 57, 94], color=color, linewidth=.65)
    arrow((outer, 94), (near, 94), color)
    text(outer - 4 if role == "replay" else outer + 4, 77,
         "state", 6.5, rotation=90)
    arrow((center, 44), (center, 35), color)

panel = FancyBboxPatch((8, 4), 222, 31,
                      boxstyle="round,pad=0,rounding_size=2",
                      linewidth=.65, edgecolor="#8695A5", facecolor="#F6F8FA")
ax.add_patch(panel)
text(W / 2, 28, "Align request / control / lifecycle traces", 7, "bold")
text(W / 2, 18, "Deadline feasibility; false accepts/rejects", 6.7)
text(W / 2, 9, "Feasible cost under the shared ledger", 6.7)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for patch, label in containers:
    assert patch.get_window_extent(renderer).contains(
        *label.get_window_extent(renderer).get_points()[0]), label.get_text()
    assert patch.get_window_extent(renderer).contains(
        *label.get_window_extent(renderer).get_points()[1]), label.get_text()
for label in labels:
    bounds = label.get_window_extent(renderer)
    assert fig.bbox.contains(*bounds.get_points()[0]), label.get_text()
    assert fig.bbox.contains(*bounds.get_points()[1]), label.get_text()
overlaps = []
for i, a in enumerate(labels):
    for b in labels[i + 1:]:
        if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
            overlaps.append([a.get_text(), b.get_text()])
assert not overlaps, overlaps

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig0_framework.pdf", metadata={"CreationDate": None, "ModDate": None})
fig.savefig(OUT / "fig0_framework.svg", metadata={"Date": None})
fig.savefig(OUT / "fig0_framework.png", dpi=300)
(QA / "FIGURE_CHECKS.json").write_text(json.dumps({
    "status": "PASS", "dimensions_pt": [W, H], "minimum_font_pt": 6.5,
    "text_labels": [t.get_text() for t in labels],
    "label_overlaps": overlaps, "labels_contained": True,
    "shared_dynamic_state": False, "physical_to_training_arrow": False,
    "formats": ["pdf", "svg", "png"],
}, indent=2) + "\n")
plt.close(fig)

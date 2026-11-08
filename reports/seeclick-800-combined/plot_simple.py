"""Simple Matplotlib plot of pooled development and test success rates."""

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scriptwob-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path(__file__).resolve().parent
with (root / "combined_levels.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

labels = [
    "L0\nAtomic action",
    "L1\nSequence",
    "L2\nConditional",
    "L3\nRepetition and\nstopping",
    "L4\nBranches\nwith loops",
    "L5\nNested\ncontrol flow",
]
rates = [float(row["success_rate_percent"]) for row in rows]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, rates, color="tab:blue", width=0.65)
ax.bar_label(bars, labels=[f"{rate:.1f}%" for rate in rates], padding=4)
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 101, 20))
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.set_ylabel("Task success rate")
ax.set_xlabel("Script complexity level")
ax.set_title("SeeClick: combined dev + test (200 tasks)")
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(root / "success_by_level.png", dpi=200)
fig.savefig(root / "success_by_level.pdf")
plt.close(fig)

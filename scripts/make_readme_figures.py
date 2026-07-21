"""Regenerate the README figures from the preserved results (results/summary.csv).

Figures A and B are reproduced directly from the aggregated run metrics.
The DC topomap is a static asset (docs/figures/fig_dc_topomap.png).
The pre-stimulus DC localization (Fig C) is produced by analysis/dc_prestimulus_test.py.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "results" / "summary.csv")

MODEL_ORDER = ["cnn", "DiffE_AT", "DiffE_mod", "shallowconvnet", "deepconvnet", "DiffE_TF", "eegnet", "mlp"]
CH = 0.20


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------- Fig A: raw vs preprocessed (all subjects) ----------------
m = df[df.source == "main_all"].copy()
m["preprocess"] = _num(m["preprocess"])
m["cond"] = m["preprocess"].map({0: "Raw", 1: "Preprocessed"})
m = m[m.cond.isin(["Raw", "Preprocessed"])]
piv = m.pivot_table(index="model", columns="cond", values="test_accuracy", aggfunc="mean")
piv = piv.reindex([x for x in MODEL_ORDER if x in piv.index])

fig, ax = plt.subplots(figsize=(9, 4.2))
x = range(len(piv))
w = 0.38
ax.bar([i - w / 2 for i in x], piv["Raw"], w, label="Raw", color="#c0392b")
ax.bar([i + w / 2 for i in x], piv["Preprocessed"], w, label="Preprocessed", color="#2c6fbb")
ax.axhline(CH, ls="--", c="gray", lw=1, label="Chance (0.20)")
ax.set_xticks(list(x))
ax.set_xticklabels(piv.index, rotation=30, ha="right")
ax.set_ylabel("Test accuracy")
ax.set_title("Raw vs. preprocessed EEG (all subjects, 5 classes)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig_raw_vs_preprocessed.png", dpi=160)
plt.close(fig)
print("saved fig_raw_vs_preprocessed.png")

# ---------------- Fig B: ablation - DC/ULF retained vs removed ----------------
a = df[df.source.isin(["ablation", "ablation_orig"])].copy()
for c in ["bandpass", "baseline"]:
    a[c] = _num(a[c])
a["dc"] = "DC / ULF removed"
a.loc[(a.bandpass == 0) & (a.baseline == 0), "dc"] = "DC / ULF retained"
g = a.groupby("dc")["test_accuracy"]
means, stds = g.mean(), g.std()
order = ["DC / ULF retained", "DC / ULF removed"]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.bar(order, [means[o] for o in order], yerr=[stds[o] for o in order],
       color=["#c0392b", "#2c6fbb"], capsize=6)
ax.axhline(CH, ls="--", c="gray", lw=1, label="Chance (0.20)")
ax.set_ylabel("Test accuracy")
ax.set_title("Ablation: the DC / ultra-low-frequency component\ndrives classification")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig_ablation_dc.png", dpi=160)
plt.close(fig)
print("saved fig_ablation_dc.png")

# ---------------- Fig C: pre-stimulus DC localization (Subject 1) ----------------
labels = ["DC\nfull trial", "DC\npre-stimulus", "DC\npost-stimulus", "DC\nbaseline-corrected"]
vals = [0.473, 0.413, 0.490, 0.267]
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(labels, vals, color=["#8e44ad", "#c0392b", "#e67e22", "#2c6fbb"])
ax.axhline(CH, ls="--", c="gray", lw=1, label="Chance (0.20)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("Accuracy (SVM-RBF, 5-fold CV)")
ax.set_ylim(0, 0.6)
ax.set_title("Where does the discriminative DC live? (Subject 1)\nIt is already in the pre-stimulus window")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig_dc_prestimulus.png", dpi=160)
plt.close(fig)
print("saved fig_dc_prestimulus.png")
print("DONE")

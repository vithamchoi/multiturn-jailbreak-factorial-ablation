#!/usr/bin/env python3
"""Generate the paper's data figures from the raw experimental artefacts.

    python3 make_figures.py <results_dir> <out_dir>

Every plotted value comes from analysis.py, which reads the result files
directly. No estimated or simulated numbers. Vector PDF sized for the Elsevier
two-column layout (3.35 in single column, 6.9 in full width).

Palette: Okabe-Ito subset, validated colourblind-safe in fixed assignment order.
Series identity is never colour-alone: every series is also labelled.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import Data, CONFIGS, RULES  # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
OUT.mkdir(parents=True, exist_ok=True)
D = Data(RES)

C = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


SHORT = ["Single-turn\ndirect", "Single-turn\nindirect", "Multi-turn\ndirect"]

# ==================================================== Fig: factorial ablation
fig, ax = plt.subplots(figsize=(5.6, 2.88))
x = np.arange(len(CONFIGS))
W = 0.34
for j, (cond, clabel) in enumerate([("baseline", "No system prompt"),
                                    ("hardened", "Safety system prompt")]):
    vals = [D.asr(c, cond) for c, _ in CONFIGS]
    los, his = zip(*[D.ci(c, cond) for c, _ in CONFIGS])
    err = np.vstack([np.array(vals) - np.array(los), np.array(his) - np.array(vals)])
    ax.bar(x + (j - 0.5) * W, vals, width=W - 0.04, color=C[j], label=clabel,
           edgecolor="white", linewidth=0.6)
    ax.errorbar(x + (j - 0.5) * W, vals, yerr=err, fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=1.8, capthick=0.8)
    for xi, v, hi in zip(x + (j - 0.5) * W, vals, his):
        ax.text(xi, hi + 0.9, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(SHORT)
ax.set_ylabel("Harmful-response rate (%)")
ax.set_ylim(0, 31)
ax.xaxis.grid(False)
F.place_legend(ax, frameon=False, handlelength=1.2, labelcolor=INK,
          borderaxespad=0.2)
tidy(ax)
fig.tight_layout()
save(fig, "fig_ablation.pdf")

# ==================================================== Fig: judge aggregation
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.40),
                         gridspec_kw={"width_ratios": [1.35, 1.0]})

ax = axes[0]
x = np.arange(len(CONFIGS))
W = 0.2
for j, (rule, rlabel) in enumerate(RULES):
    vals = [D.asr(c, "baseline", rule) for c, _ in CONFIGS]
    ax.bar(x + (j - 1.5) * W, vals, width=W - 0.03, color=C[j % len(C)],
           label=rlabel.replace("~", " "), edgecolor="white", linewidth=0.6)
    for xi, v in zip(x + (j - 1.5) * W, vals):
        if v >= 1.0:
            ax.text(xi, v + 0.7, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.4, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(SHORT)
ax.set_ylabel("Harmful-response rate (%)")
ax.set_ylim(0, 34)
ax.xaxis.grid(False)
F.place_legend(ax, frameon=False,
          handlelength=1.2, labelcolor=INK, borderaxespad=0.1, ncol=4,
          columnspacing=0.8, fontsize=6.5)
ax.set_title("(a) The reported rate depends on the aggregation rule",
             loc="left", pad=4)
tidy(ax)

ax = axes[1]
ja = D.judge_agreement()
levels = ja["levels"]
M = np.zeros((len(levels), len(levels)))
for (a, b), v in ja["confusion"].items():
    M[levels.index(a), levels.index(b)] = v
im = ax.imshow(np.log10(M + 1), cmap="Blues", aspect="auto")
for i in range(len(levels)):
    for k in range(len(levels)):
        v = int(M[i, k])
        ax.text(k, i, f"{v}", ha="center", va="center", fontsize=7,
                color="white" if np.log10(v + 1) > 1.9 else INK)
ax.set_xticks(range(len(levels)))
ax.set_yticks(range(len(levels)))
ax.set_xticklabels(levels)
ax.set_yticklabels(levels)
ax.set_xlabel("Judge 2 score (Qwen3-32B)")
ax.set_ylabel("Judge 1 score (Llama-3.3-70B)")
ax.grid(False)
ax.set_title(rf"(b) Judge confusion, $\kappa={ja['kappa']:.3f}$", loc="left", pad=4)
tidy(ax)

fig.tight_layout(w_pad=2.2)
save(fig, "fig_judges.pdf")

# ==================================================== Fig: measurement layers
kw = D.keyword_layer()
a3 = CONFIGS[2][0]
labels = ["Keyword detector\n(single-turn)", "Keyword detector\n(multi-turn)",
          "Rubric judges, OR\n(multi-turn)", "Rubric judges, AND\n(multi-turn)"]
vals = [kw["single_pct"], kw["multi_pct"],
        D.asr(a3, "baseline", "or"), D.asr(a3, "baseline", "and")]
cis = [kw["single_ci"], kw["multi_ci"],
       D.ci(a3, "baseline", "or"), D.ci(a3, "baseline", "and")]
cols = [C[3], C[1], C[0], C[2]]

fig, ax = plt.subplots(figsize=(5.6, 2.93))
y = np.arange(len(labels))[::-1]
err = np.vstack([[v - lo for v, (lo, hi) in zip(vals, cis)],
                 [hi - v for v, (lo, hi) in zip(vals, cis)]])
ax.barh(y, vals, height=0.6, color=cols, edgecolor="white", linewidth=0.6)
ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=INK, elinewidth=0.8,
            capsize=1.8, capthick=0.8)
for yi, v in zip(y, vals):
    ax.text(v + 1.8, yi, f"{v:.2f}%", va="center", ha="left", fontsize=7.2,
            color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Reported attack success rate (%)")
ax.set_xlim(0, 118)
ax.yaxis.grid(False)
tidy(ax)
fig.tight_layout()
save(fig, "fig_layers.pdf")

# ==================================================== Fig: category breakdown
cats = D.categories()
order = sorted(cats, key=lambda c: -(D.cat_asr(c, CONFIGS[1][0])[0] or 0))
fig, ax = plt.subplots(figsize=(5.6, 2.99))
y = np.arange(len(order))[::-1]
W = 0.26
for j, (cfg, _) in enumerate(CONFIGS):
    vals = [D.cat_asr(c, cfg)[0] for c in order]
    ax.barh(y + (1 - j) * W, vals, height=W - 0.03, color=C[j],
            edgecolor="white", linewidth=0.6,
            label=["Single, direct", "Single, indirect", "Multi, direct"][j])
ax.set_yticks(y)
ax.set_yticklabels([c.replace("/", " /\n") for c in order])
ax.set_xlabel("Harmful-response rate (%)")
ax.set_xlim(0, 47)
ax.yaxis.grid(False)
F.place_legend(ax, frameon=False, handlelength=1.2, labelcolor=INK,
          borderaxespad=0.3)
tidy(ax)
fig.tight_layout()
save(fig, "fig_category.pdf")

print("\nAll figures generated from:", RES.resolve())

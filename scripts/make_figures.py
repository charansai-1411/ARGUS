"""Phase 6: results & figures pack for the report and paper.

    python scripts/make_figures.py
Writes docs/results/*.png (200 dpi, light, print-ready) and docs/results/RESULTS.md.
Reads outputs/*.json from earlier phases; the full-HI-Small GNN numbers come from the Kaggle
run recorded in the README (that run's JSON stayed on Kaggle).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from argus.config import resolve  # noqa: E402

OUT = resolve("docs/results")
# Reference palette, light mode (validated: all-pairs CVD dE 9.2, normal-vision 24.0).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"GNN (GINE)": "#2a78d6", "GNN": "#2a78d6", "LightGBM": "#eb6834", "Rules": "#1baf7a"}
GNN_HI_SMALL = {"pr_auc": 0.7258, "f1": 0.7291, "precision": 0.8607, "recall": 0.6324,
                "ring_recall@1": 0.604, "ring_recall@0.5": 0.901}   # Kaggle T4, time-respecting, epoch 9

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    p = resolve(f"outputs/{name}")
    return json.loads(p.read_text()) if p.exists() else None


def clean_axis(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_hi_small(base):
    rows = {"Rules": base["rules"], "LightGBM": base["lightgbm"], "GNN (GINE)": GNN_HI_SMALL}
    metrics = [("pr_auc", "PR-AUC"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    names = list(rows)
    for ax, (k, title) in zip(axes, metrics):
        vals = [rows[n][k] for n in names]
        ax.barh(names, vals, color=[COLOR[n] for n in names], height=0.6, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1.15); ax.set_title(title, loc="left")  # list order bottom->top: GNN on top
        clean_axis(ax); ax.set_xticks([0, 0.5, 1])
    fig.suptitle("Detection on IBM AML HI-Small (5.08M transfers; test = last 20% by time)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    save(fig, "fig1_hi_small_detection.png")


def fig_attack_D(lgb, gnn):
    ops = list(lgb["single"])
    order = sorted(ops, key=lambda o: -max(lgb["single"][o]["D"], gnn["single"][o]["D"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    h = 0.38
    for j, (name, res) in enumerate((("LightGBM", lgb), ("GNN", gnn))):
        vals = [res["single"][o]["D"] for o in order]
        ys = [i + (j - 0.5) * h for i in range(len(order))]
        ax.barh(ys, vals, height=h, color=COLOR[name], label=name, edgecolor=SURFACE, linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(max(v, 0) + 0.01, y, f"{v:+.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order]); ax.invert_yaxis()
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlim(-0.05, 0.6); ax.set_xlabel("D = F1 clean − F1 under attack (lower is better)")
    ax.set_title("Accuracy lost to each attack on the multi-bank twin", loc="left")
    ax.legend(frameon=False, loc="lower right"); clean_axis(ax)
    save(fig, "fig2_attack_degradation.png")


def fig_adaptive(lgb, gnn):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for name, res in (("LightGBM", lgb), ("GNN", gnn)):
        xs, ys = [0], [res["clean"]["f1"]]
        for r in res["adaptive"]:
            xs += [2 * r["round"] - 1, 2 * r["round"]]
            ys += [r["attack"]["f1"], r["retrained"]["f1"]]
        ax.plot(xs, ys, color=COLOR[name], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ax.text(xs[-1] + 0.15, ys[-1], f"{name} {ys[-1]:.3f}", va="center", fontsize=9, color=INK)
        for r in res["adaptive"]:
            ax.annotate(r["added"].replace("_", " "), (2 * r["round"] - 1, r["attack"]["f1"]),
                        textcoords="offset points", xytext=(0, -14 if name == "LightGBM" else 9),
                        ha="center", fontsize=7.5, color=INK2)
    n = len(lgb["adaptive"])
    ticks = ["clean"] + [s for t in range(1, n + 1) for s in (f"R{t}\nattacked", f"R{t}\nretrained")]
    ax.set_xticks(range(2 * n + 1), ticks, fontsize=8.5)
    ax.set_ylim(0.4, 1.02); ax.set_xlim(-0.3, 2 * n + 1.6); ax.set_ylabel("Test F1")
    ax.set_title("Adaptive attacker: each round adds the tactic that hurts the current model most",
                 loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2); clean_axis(ax, "y")
    save(fig, "fig3_adaptive_rounds.png")


def fig_agent(agent):
    conds = [c for c in ("camouflage", "split", "clean") if c in agent]   # bottom->top: clean on top
    metrics = [("mean_precision", "Account precision"), ("mean_recall", "Account recall"),
               ("mean_jaccard", "Jaccard"), ("entry_identified", "Victim identified")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.4), sharey=True)
    for ax, (k, title) in zip(axes, metrics):
        vals = [agent[c][k] for c in conds]
        ax.barh(conds, vals, color=COLOR["GNN"], height=0.6, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1.18); ax.set_title(title, loc="left")
        ax.set_xticks([0, 0.5, 1]); clean_axis(ax)
    fig.suptitle("Agent ring reconstruction vs twin ground truth (103 test rings, by attack on the twin)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.06)
    save(fig, "fig4_agent_reconstruction.png")


def tables(base, lgb, gnn, agent):
    L = ["# ARGUS — results pack", "",
         "All numbers are produced by the scripts in this repo; figures in this folder are generated by "
         "`scripts/make_figures.py`. Twin results are a single seed.", "",
         "## Table 1 — Detection on IBM AML HI-Small (test = last 20% by time)", "",
         "| Model | PR-AUC | F1 | Precision | Recall | Ring recall @1 | Ring recall @0.5 |", "|---|---|---|---|---|---|---|"]
    for name, r in (("Rules", base["rules"]), ("LightGBM", base["lightgbm"]), ("GNN (GINE, time-respecting)", GNN_HI_SMALL)):
        L.append(f"| {name} | {r['pr_auc']:.3f} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                 f"{r.get('ring_recall@1', float('nan')):.3f} | {r.get('ring_recall@0.5', float('nan')):.3f} |")
    L += ["", "## Table 2 — Single-operator attacks on the twin (D = degradation, R = recovery, F1)", "",
          "| Attack | LightGBM attacked | LightGBM retrained | LightGBM D | LightGBM R | GNN attacked | GNN retrained | GNN D | GNN R |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| *(clean)* | {lgb['clean']['f1']:.3f} | — | — | — | {gnn['clean']['f1']:.3f} | — | — | — |"]
    for op in lgb["single"]:
        a, b = lgb["single"][op], gnn["single"][op]
        L.append(f"| {op} | {a['attack']['f1']:.3f} | {a['retrained']['f1']:.3f} | {a['D']:+.3f} | {a['R']:+.3f} | "
                 f"{b['attack']['f1']:.3f} | {b['retrained']['f1']:.3f} | {b['D']:+.3f} | {b['R']:+.3f} |")
    for name, res in (("LightGBM", lgb), ("GNN", gnn)):
        L += ["", f"## Table 3{'a' if name == 'LightGBM' else 'b'} — Adaptive attacker vs {name}", "",
              "| Round | Tactic added | F1 before | F1 attacked | F1 retrained | D | R |", "|---|---|---|---|---|---|---|"]
        for r in res["adaptive"]:
            L.append(f"| {r['round']} | {r['added']} | {r['before_f1']:.3f} | {r['attack']['f1']:.3f} | "
                     f"{r['retrained']['f1']:.3f} | {r['D']:+.3f} | {r['R']:+.3f} |")
    L += ["", "## Table 4 — Agent ring reconstruction (103 test rings, LightGBM scores)", "",
          "| Twin | Rings detected | Precision | Recall | Jaccard | Victim identified |", "|---|---|---|---|---|---|"]
    for c, v in agent.items():
        L.append(f"| {c} | {v['rings_detected']}/{v['test_rings']} | {v['mean_precision']:.3f} | {v['mean_recall']:.3f} | "
                 f"{v['mean_jaccard']:.3f} | {v['entry_identified']:.3f} |")
    L += ["", "## Figures", "", "![](fig1_hi_small_detection.png)", "", "![](fig2_attack_degradation.png)", "",
          "![](fig3_adaptive_rounds.png)", "", "![](fig4_agent_reconstruction.png)", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / "RESULTS.md")


def main():
    base, lgb, gnn = load("baselines_HI-Small.json"), load("twin_lightgbm.json"), load("twin_gnn.json")
    agent = {k: load(f"agent_eval_lightgbm{s}.json") for k, s in (("clean", ""), ("split", "_split"), ("camouflage", "_camouflage"))}
    agent = {k: v for k, v in agent.items() if v}
    fig_hi_small(base)
    fig_attack_D(lgb, gnn)
    fig_adaptive(lgb, gnn)
    fig_agent(agent)
    tables(base, lgb, gnn, agent)


if __name__ == "__main__":
    main()

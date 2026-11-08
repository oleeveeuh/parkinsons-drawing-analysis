#!/usr/bin/env python3
"""Portfolio-site story figures, rendered from this repo's frozen results
JSONs only (results/ — git-ignored; regenerate via the README commands).
Fieldnotes theme: the palette + fonts of oleeveeuh.github.io, so dialog
figures read as one system with the gallery card. Every number a figure
draws is read from a frozen artifact at render time — nothing is typed in.

Usage: python scripts/portfolio_figures.py [--out DIR] [--scale 2]
Requires: pip install -e ".[figures]" and the site's static TTFs
(Source Sans 3 400/600/700, IBM Plex Mono 400/500) in ~/Library/Fonts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---- the fieldnotes theme (oleeveeuh.github.io) ---------------------------
PAPER = "#fbf9f4"       # cream card surface — story-figure background
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#77776c"
GRID = "#d8d1c4"
EDGE = "#c9c2b2"
SAGE = "#89927a"        # positive / the corrected protocol
SAGE_LIGHT = "#dfe1d5"
WINE = "#482728"        # failure voice: leaks, collapsed stacks, dead claims
CRIMSON = "#a43b36"     # site red accent
ROSE = "#d4708c"        # challenger voice: deep features, MoE, image modality
SUMI = "#1E342E"        # neutral baselines

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans 3", "IBM Plex Sans", "DejaVu Sans"],
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def check_fonts() -> None:
    """matplotlib silently falls back to DejaVu Sans — refuse to render
    the house style on a machine without the installed static TTFs."""
    from matplotlib import font_manager as fm
    for family in ("Source Sans 3", "IBM Plex Mono"):
        try:
            path = fm.findfont(fm.FontProperties(family=family),
                               fallback_to_default=False)
        except ValueError as exc:
            raise RuntimeError(
                f"{family} is not installed — the fieldnotes figures require "
                "the site's static TTFs in ~/Library/Fonts") from exc
        print(f"font ok: {family} -> {Path(path).name}")


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def metrics(rel: str, *path: str) -> dict:
    """participant_metrics for one model; path = [task,] model."""
    data = load(rel)
    node = data
    for key in path[:-1]:
        node = node["tasks" if key in ("spiral", "wave") else "models"][key]
    return node["models"][path[-1]]["participant_metrics"]


def ci_yerr(m: dict) -> list:
    acc = m["accuracy"]
    lo, hi = m["subject_bootstrap_ci"]["accuracy"]
    return [[acc - lo], [hi - acc]]


def label_bar(ax, bar, value, m=None, inside_threshold=0.55):
    """White label inside tall bars; above the CI whisker for short ones."""
    if value >= inside_threshold:
        ax.text(bar.get_x() + bar.get_width() / 2, value - 0.055,
                f"{value:.3f}", ha="center", va="top", fontsize=9.5,
                color=PAPER, fontweight="bold", zorder=6)
    else:
        top = value
        if m is not None:
            top = m["subject_bootstrap_ci"]["accuracy"][1]
        ax.text(bar.get_x() + bar.get_width() / 2, top + 0.018,
                f"{value:.3f}", ha="center", va="bottom", fontsize=9.5,
                color=INK2, zorder=6)


def mono_note(fig, text: str) -> None:
    fig.text(0.01, -0.03, text, fontsize=8.2, color=MUTED,
             ha="left", va="top", fontfamily="IBM Plex Mono", wrap=True)


# ---------------------------------------------------------------- figure 1
def audit_units(out: Path) -> None:
    audit = load("results/data_audit.json")
    spatial = audit["spatial"]["summary"]["per_task"]
    temporal = audit["temporal"]["summary"]
    leak = {row["task"]: row for row in audit["spatial"]["vendor_split_leakage"]}

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(8.6, 4.3), width_ratios=[1.45, 1.0])
    fig.subplots_adjust(wspace=0.28, bottom=0.16)

    # A: advertised samples vs actual participants
    groups = [
        ("temporal\n(UCI)", temporal["n_recordings"], temporal["n_subjects"]),
        ("spiral\n(Kaggle)", spatial["spiral"]["n_images"], spatial["spiral"]["n_subjects"]),
        ("wave\n(Kaggle)", spatial["wave"]["n_images"], spatial["wave"]["n_subjects"]),
    ]
    x = np.arange(len(groups))
    w = 0.36
    for i, (label, samples, participants) in enumerate(groups):
        hollow = ax_a.bar(i - w / 2, samples, width=w, facecolor="none",
                          edgecolor=MUTED, linewidth=1.6, zorder=3)
        solid = ax_a.bar(i + w / 2, participants, width=w, color=SAGE, zorder=3)
        ax_a.text(i - w / 2, samples + 2, str(samples), ha="center", va="bottom",
                  fontsize=10, color=INK2)
        ax_a.text(i + w / 2, participants + 2, str(participants), ha="center",
                  va="bottom", fontsize=10, color=INK2, fontweight="bold")
        if samples != participants:
            ax_a.annotate(f"{samples / participants:.1f}×", xy=(i - w / 2, samples / 2),
                          ha="center", va="center", fontsize=11, color=WINE,
                          fontweight="bold")
    ax_a.set_xticks(x, [g[0] for g in groups], fontsize=9.5)
    ax_a.set_ylabel("count")
    ax_a.set_ylim(0, 132)
    ax_a.grid(axis="x", visible=False)
    ax_a.set_title("advertised samples vs actual participants",
                   fontsize=10.5, color=INK2, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=MUTED,
                             linewidth=1.6),
               plt.Rectangle((0, 0), 1, 1, facecolor=SAGE)]
    ax_a.legend(handles, ["samples the dataset advertises", "actual participants"],
                loc="upper right", fontsize=8.5, frameon=False)

    # B: drawings per participant — who contributes the images
    dist = spatial["spiral"]["drawings_per_subject"]
    xs = sorted(int(k) for k in dist)
    heights = [dist[str(k)] for k in xs]
    colors = [SAGE if k == 3 else WINE for k in xs]
    bars = ax_b.bar(xs, heights, width=1.1, color=colors, zorder=3)
    for bar, k, h in zip(bars, xs, heights):
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                  f"{h} participant{'s' if h > 1 else ''} × {k}",
                  ha="center", va="bottom", fontsize=9, color=INK2)
    ax_b.set_xticks(xs)
    ax_b.set_xlabel("drawings per participant (per task)")
    ax_b.set_ylabel("participants")
    ax_b.set_xlim(0, 17.5)
    ax_b.set_ylim(0, 31)
    ax_b.grid(axis="x", visible=False)
    ax_b.set_title("who contributes the 102 images",
                   fontsize=10.5, color=INK2, loc="left")
    extra = sum(int(k) * v for k, v in dist.items() if int(k) > 3)
    ax_b.text(9.5, 20, f"two people contribute\n{extra} of 102 images per task",
              ha="center", va="center", fontsize=9.5, color=WINE, linespacing=1.35)

    leak_spiral = leak["spiral"]
    leak_wave = leak["wave"]
    mono_note(fig, f"supplied test split leaks every test participant into training: "
                   f"spiral {leak_spiral['shared_subjects']}/{leak_spiral['test_subjects']}, "
                   f"wave {leak_wave['shared_subjects']}/{leak_wave['test_subjects']} "
                   f"({leak_spiral['test_subjects_leaked']}) · source: results/data_audit.json, "
                   "docs/data_audit.md")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def results_ci(out: Path) -> None:
    tree_key = "decision_tree_depth2_4"
    temporal = {
        "dummy": metrics("results/real/temporal_real.json", "dummy_majority"),
        "tree": metrics("results/real/temporal_real.json", tree_key),
        "logreg": metrics("results/real/temporal_real.json", "logreg_balanced"),
    }
    spiral = {
        "dummy": metrics("results/real/spatial_real_hog_shape.json", "spiral", "dummy_majority"),
        "tree": metrics("results/real/spatial_real_hog_shape.json", "spiral", tree_key),
        "logreg": metrics("results/real/spatial_real_hog_shape.json", "spiral", "logreg_balanced"),
    }
    wave = {
        "dummy": metrics("results/real/spatial_real_hog_shape.json", "wave", "dummy_majority"),
        "tree": metrics("results/real/spatial_real_hog_shape.json", "wave", tree_key),
        "logreg": metrics("results/real/spatial_real_hog_shape.json", "wave", "logreg_balanced"),
    }
    groups = [("temporal · UCI\nn = 40 subjects", temporal),
              ("spiral · Kaggle\nn = 28 subjects", spiral),
              ("wave · Kaggle\nn = 28 subjects", wave)]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(groups))
    w = 0.24
    slots = [("dummy", -w, SUMI), ("tree", 0.0, MUTED), ("logreg", w, SAGE)]
    for i, (label, models) in enumerate(groups):
        for key, off, color in slots:
            m = models[key]
            bar = ax.bar(i + off, m["accuracy"], width=w, color=color, zorder=3,
                         yerr=ci_yerr(m), error_kw=dict(ecolor=INK2, lw=1.1, capsize=3))
            label_bar(ax, bar[0], m["accuracy"], m, inside_threshold=0.72)
            auc = m.get("roc_auc")
            if auc is not None:
                # inside the bar, under the value label — keeps the band
                # above the whiskers free for the legend
                ax.text(i + off, m["accuracy"] - 0.115, f"auc {auc:.3f}",
                        ha="center", va="top", fontsize=7.4, color=PAPER,
                        fontfamily="IBM Plex Mono", zorder=6)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=9.5)
    ax.set_ylabel("participant-level accuracy (out-of-fold)")
    ax.set_ylim(0, 1.06)
    ax.grid(axis="x", visible=False)
    ax.set_title("subject-grouped nested CV — the single-model family, accuracy with 95% CI",
                 fontsize=10.5, color=INK2, loc="left", pad=30)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for _, _, c in slots]
    ax.legend(handles, ["majority-class baseline", "decision tree (depth 2–4)",
                        "logistic regression (class-weighted)"],
              loc="lower left", bbox_to_anchor=(0.0, 1.012), ncols=3,
              fontsize=8.8, frameon=False, columnspacing=1.2, handlelength=1.1)
    mono_note(fig, "identical participant folds for spiral & wave — paired sign-flip "
                   "difference 0.000, p = 1.0 (28 pairs, 10,000 permutations) · CI: 2,000 "
                   "deterministic subject-bootstrap resamples · xgboost is wired as an "
                   "optional extra but has no run in the frozen artifacts · sources: "
                   "results/real/temporal_real.json, results/real/spatial_real_hog_shape.json")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def split_impact(out: Path) -> None:
    impact = load("results/impact_numbers.json")["spatial_split_impact"]
    leaky = impact["leaky_image_level_split"]
    grouped = impact["participant_grouped_split"]
    diff = impact["difference_leaky_minus_grouped"]
    paired = load("results/real/spatial_real_hog_shape.json")[
        "task_comparison"]["paired_sign_flip_accuracy"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.8, 4.3),
                                     width_ratios=[1.0, 1.15])
    fig.subplots_adjust(wspace=0.30, bottom=0.14)

    # A: same model, only the split differs
    means = [leaky["participant_accuracy_mean"], grouped["participant_accuracy_mean"]]
    stds = [leaky["participant_accuracy_std"], grouped["participant_accuracy_std"]]
    bars = ax_a.bar([0, 1], means, width=0.55, color=[WINE, SAGE], zorder=3,
                    yerr=[[s] for s in stds],
                    error_kw=dict(ecolor=INK2, lw=1.3, capsize=5))
    ax_a.text(0, means[0] - 0.10, f"{means[0]:.3f} ± {stds[0]:.3f}", ha="center",
              va="top", fontsize=10, color=PAPER, fontweight="bold", zorder=6)
    ax_a.text(1, means[1] - 0.10, f"{means[1]:.3f} ± {stds[1]:.3f}", ha="center",
              va="top", fontsize=10, color=PAPER, fontweight="bold", zorder=6)
    ax_a.set_xticks([0, 1], ["leaky\nimage-level split", "participant-\ngrouped split"],
                    fontsize=9.5)
    ax_a.set_ylabel("participant accuracy (mean of 30 seeds)")
    ax_a.set_ylim(0, 1.0)
    ax_a.grid(axis="x", visible=False)
    ax_a.set_title("same model, same aggregation — only the split differs",
                   fontsize=10.5, color=INK2, loc="left")

    # B: the finding the leak manufactured — and erased
    claim_pp = 9.6   # v1-era claim, quoted in the committed README's comparison section
    measured_pp = 100 * paired["mean_difference"]
    ax_b.hlines([1, 0], 0, [claim_pp, measured_pp], color=[WINE, SAGE],
                lw=2.4, zorder=3)
    ax_b.scatter([claim_pp], [1], s=90, color=WINE, zorder=4)
    ax_b.scatter([measured_pp], [0], s=90, color=SAGE, zorder=4)
    ax_b.text(claim_pp + 0.25, 1.0,
              f"+{claim_pp:.1f} pp\nv1 claim, image-level era (p = 0.212)",
              va="center", fontsize=9.5, color=WINE, linespacing=1.35)
    ax_b.text(measured_pp + 0.25, 0.0,
              f"{measured_pp:.3f} pp\nmeasured under grouped paired CV "
              f"(p = {paired['p_value']:.1f})", va="center", fontsize=9.5,
              color=INK2, linespacing=1.35)
    ax_b.set_yticks([1, 0], ["v1 pipeline\nclaim", "measured,\ngrouped & paired"],
                    fontsize=9.5)
    ax_b.set_xlabel("spiral − wave participant-accuracy gap (pp)")
    ax_b.set_xlim(-0.4, 14.5)
    ax_b.set_ylim(-0.6, 1.6)
    ax_b.grid(axis="y", visible=False)
    ax_b.set_title("the task-complexity finding the leak manufactured",
                   fontsize=10.5, color=INK2, loc="left")
    ax_b.text(0.985, 0.06,
              f"paired sign-flip test, {paired['n_pairs']} pairs, "
              f"{paired['n_permutations']:,} permutations",
              transform=ax_b.transAxes, ha="right", fontsize=8.2, color=MUTED,
              fontfamily="IBM Plex Mono")

    mono_note(fig, f"mean delta {100 * diff['mean']:+.1f} pp, leaky wins "
                   f"{diff['leaky_wins_seeds']} seeds (per-seed range "
                   f"{100 * diff['range'][0]:+.1f} .. {100 * diff['range'][1]:+.1f} pp) · "
                   f"unit inflation: {leaky['advertised_n']} advertised images were "
                   f"{leaky['true_participant_n']} people (3.6×) · "
                   f"{int(round(100 * leaky['subject_in_training_exposure']))}–100% of test "
                   "images had their subject in training · source: results/impact_numbers.json")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def deep_features(out: Path) -> None:
    hog = {t: metrics("results/real/spatial_real_hog_shape.json", t, "logreg_balanced")
           for t in ("spiral", "wave")}
    cnn = {t: metrics("results/real/spatial_real_cnn_hog_shape.json", t, "logreg_balanced")
           for t in ("spiral", "wave")}

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(2)
    w = 0.34
    deltas = {"spiral": "+1 participant", "wave": "−2 participants"}
    for i, task in enumerate(("spiral", "wave")):
        for off, m, color in ((-w / 2, hog[task], SUMI), (w / 2, cnn[task], ROSE)):
            bar = ax.bar(i + off, m["accuracy"], width=w, color=color, zorder=3,
                         yerr=ci_yerr(m), error_kw=dict(ecolor=INK2, lw=1.2, capsize=4))
            label_bar(ax, bar[0], m["accuracy"], m)
            hi = m["subject_bootstrap_ci"]["accuracy"][1]
            ax.text(i + off, hi + 0.03, f"auc {m['roc_auc']:.3f}", ha="center",
                    va="bottom", fontsize=8.2, color=INK2, fontfamily="IBM Plex Mono")
            if color is ROSE:  # direct-label the participant delta on the cnn bars
                ax.text(i + off, m["accuracy"] - 0.115, deltas[task], ha="center",
                        va="top", fontsize=8.6, color=PAPER, fontweight="bold",
                        zorder=6)
    ax.set_xticks(x, ["spiral · n = 28 subjects", "wave · n = 28 subjects"], fontsize=9.5)
    ax.set_ylabel("participant-level accuracy (out-of-fold)")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.set_title("frozen deep features, measured — identical protocol, identical folds",
                 fontsize=10.5, color=INK2, loc="left", pad=30)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SUMI),
               plt.Rectangle((0, 0), 1, 1, facecolor=ROSE)]
    ax.legend(handles, ["HOG + shape descriptors", "frozen ResNet-18 + HOG + shape"],
              loc="lower left", bbox_to_anchor=(0.0, 1.012), ncols=2,
              fontsize=8.8, frameon=False, columnspacing=1.2, handlelength=1.1)
    mono_note(fig, "512-d frozen ImageNet ResNet-18 embeddings (torchvision weights, eval "
                   "mode, hashed cache) prepended to HOG+shape · 95% CIs overlap almost "
                   "completely — no evidence frozen deep features help or hurt at n = 28 · "
                   "not a trained deep model · sources: results/real/spatial_real_"
                   "cnn_hog_shape.json, results/real/spatial_real_hog_shape.json")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fusion(out: Path) -> None:
    syn = {k: metrics("results/fusion/fusion_paired_synthetic.json", k)
           for k in ("stacked_fusion", "single_temporal", "single_image",
                     "early_fusion", "late_fusion")}
    abl = {k: metrics("results/fusion/fusion_feature_ablation_real.json", k)
           for k in ("single_shape", "late_fusion", "stacked_fusion",
                     "single_hog", "early_fusion")}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 4.5))
    fig.subplots_adjust(wspace=0.26, bottom=0.18)

    order_a = [("stacked\nmeta-learner", "stacked_fusion", WINE),
               ("recordings\nonly", "single_temporal", SUMI),
               ("drawings\nonly", "single_image", ROSE),
               ("early\n(concat)", "early_fusion", MUTED),
               ("late\n(probs)", "late_fusion", SAGE)]
    for i, (label, key, color) in enumerate(order_a):
        m = syn[key]
        bar = ax_a.bar(i, m["accuracy"], width=0.62, color=color, zorder=3)
        label_bar(ax_a, bar[0], m["accuracy"], m)
    ax_a.set_xticks(range(len(order_a)), [o[0] for o in order_a], fontsize=8.8)
    ax_a.set_ylim(0, 1.06)
    ax_a.grid(axis="x", visible=False)
    ax_a.set_ylabel("participant accuracy (out-of-fold)")
    ax_a.set_title("synthetic paired modalities — signal split\nacross modalities (n = 12)",
                   fontsize=10, color=INK2, loc="left")
    ax_a.annotate("the only architecture\nabove both singles", xy=(4, 0.96),
                  xytext=(2.6, 0.86), fontsize=8.8, color=INK2, ha="center",
                  arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2),
                  linespacing=1.3)

    order_b = [("shape\nonly", "single_shape", SUMI),
               ("late\n(probs)", "late_fusion", WINE),
               ("stacked\nmeta-learner", "stacked_fusion", WINE),
               ("hog\nonly", "single_hog", SAGE),
               ("early\n(concat)", "early_fusion", MUTED)]
    for i, (label, key, color) in enumerate(order_b):
        m = abl[key]
        bar = ax_b.bar(i, m["accuracy"], width=0.62, color=color, zorder=3)
        label_bar(ax_b, bar[0], m["accuracy"], m)
    ax_b.set_xticks(range(len(order_b)), [o[0] for o in order_b], fontsize=8.8)
    ax_b.set_ylim(0, 1.06)
    ax_b.grid(axis="x", visible=False)
    ax_b.set_title("real feature-family ablation — same 28\nparticipants, same drawings",
                   fontsize=10, color=INK2, loc="left")
    ax_b.annotate("fusing never beat the stronger\nsingle feature family",
                  xy=(2.05, 0.775), xytext=(1.5, 0.965), fontsize=8.8,
                  color=INK2, ha="center", va="top",
                  arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))

    mono_note(fig, "early = concatenated feature vectors → one classifier · late = per-modality "
                   "probabilities averaged (mixing weight picked on inner folds only) · stacked "
                   "= meta-learner on inner-fold OOF probabilities · sources: results/fusion/"
                   "fusion_paired_synthetic.json, results/fusion/fusion_feature_ablation_real.json")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def moe(out: Path) -> None:
    real = {k: metrics("results/moe/moe_task_experts_real.json", k)
            for k in ("single_wave", "pooled_logreg", "moe_task_experts", "single_spiral")}
    gate = load("results/moe/moe_task_experts_real.json")["gate_mean_weights"]
    syn = load("results/moe/moe_synthetic.json")
    val = load("results/moe/moe_validate_large.json")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 4.5),
                                     width_ratios=[1.15, 1.0])
    fig.subplots_adjust(wspace=0.26, bottom=0.18)

    order_a = [("wave\nexpert", "single_wave", MUTED),
               ("pooled\nlogreg", "pooled_logreg", SUMI),
               ("mixture of\nexperts", "moe_task_experts", ROSE),
               ("spiral\nexpert", "single_spiral", SAGE)]
    for i, (label, key, color) in enumerate(order_a):
        m = real[key]
        bar = ax_a.bar(i, m["accuracy"], width=0.62, color=color, zorder=3,
                       yerr=ci_yerr(m), error_kw=dict(ecolor=INK2, lw=1.1, capsize=3))
        label_bar(ax_a, bar[0], m["accuracy"], m)
    ax_a.set_xticks(range(len(order_a)), [o[0] for o in order_a], fontsize=8.8)
    ax_a.set_ylim(0, 1.02)
    ax_a.grid(axis="x", visible=False)
    ax_a.set_ylabel("participant accuracy (out-of-fold)")
    ax_a.set_title("real data — per-task experts, same 28\nparticipants: the mixture adds nothing",
                   fontsize=10, color=INK2, loc="left")

    # B: the architecture is data-hungry (synthetic two-regime fixtures).
    # Panel uses participant-level OOF accuracy for the CV fixture and the
    # holdout test accuracy for the n=600 run — as reported in the artifacts.
    moe_syn = syn["models"]["moe_em"]["participant_metrics"]["accuracy"]
    best_syn = max(v["participant_metrics"]["accuracy"]
                   for k, v in syn["models"].items() if k != "moe_em")
    moe_val = val["models"]["moe_em"]["test_accuracy"]
    best_val = max(v["test_accuracy"]
                   for k, v in val["models"].items() if k != "moe_em")
    weak_val = val["models"]["single_logreg_strong_l2"]["test_accuracy"]
    purity_syn = syn["gate_localisation"]["gate_regime_purity"]
    purity_val = val["gate_regime_purity"]
    x = np.arange(2)
    w = 0.34
    for i, (moe_acc, best_acc) in enumerate(((moe_syn, best_syn), (moe_val, best_val))):
        ax_b.bar(i - w / 2, moe_acc, width=w, color=ROSE, zorder=3)
        ax_b.text(i - w / 2, moe_acc + 0.018, f"{moe_acc:.3f}", ha="center",
                  fontsize=9.5, color=INK2, fontweight="bold")
        ax_b.bar(i + w / 2, best_acc, width=w, color=SAGE, zorder=3)
        ax_b.text(i + w / 2, best_acc + 0.018, f"{best_acc:.3f}", ha="center",
                  fontsize=9.5, color=INK2, fontweight="bold")
    ax_b.text(0.5, 0.985,
              f"beats its weak expert ({weak_val:.3f}) — never the best single",
              transform=ax_b.transAxes, ha="center", va="top", fontsize=8.4,
              color=INK2, fontfamily="IBM Plex Mono")
    ax_b.set_xticks(x, [f"two-regime fixture · n = 24\ngate purity {purity_syn:.3f}",
                        f"n = 600 subjects (1,200 rows)\ngate purity {purity_val:.3f}"],
                    fontsize=8.4)
    ax_b.set_ylim(0, 1.0)
    ax_b.grid(axis="x", visible=False)
    ax_b.set_title("synthetic two-regime fixtures —\nthe gate localises; the mixture still loses",
                   fontsize=10, color=INK2, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ROSE),
               plt.Rectangle((0, 0), 1, 1, facecolor=SAGE)]
    ax_b.legend(handles, ["mixture of experts", "best single model"],
                loc="center left", bbox_to_anchor=(0.38, 0.50),
                fontsize=8.6, frameon=False)

    mono_note(fig, f"softmax-gated MoE, EM training, k-means gate initialisation (uniform init "
                   f"collapses onto one expert), per-expert feature slices · real-data gate "
                   f"weights {gate[0]:.3f} / {gate[1]:.3f} (near-uniform) · sources: "
                   f"results/moe/moe_task_experts_real.json, results/moe/moe_synthetic.json, "
                   f"results/moe/moe_validate_large.json")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="render scale (2 = retina for the site)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    check_fonts()
    if args.scale != 1.0:
        plt.rcParams["figure.dpi"] = 150 * args.scale

    jobs = {
        "pd-drawings-audit-units.png": audit_units,
        "pd-drawings-results-ci.png": results_ci,
        "pd-drawings-split-impact.png": split_impact,
        "pd-drawings-deep-features.png": deep_features,
        "pd-drawings-fusion.png": fusion,
        "pd-drawings-moe.png": moe,
    }
    for name, fn in jobs.items():
        fn(args.out / name)
        print(f"written: {args.out / name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate all visualization plots for the CS181 poker project.

Reads CSV data from logs/viz_data/ and saves PNG figures to figures/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

DATA_DIR = PROJECT / "logs" / "viz_data"
FIG_DIR = PROJECT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
LABELS = ["Weak", "Mid", "Strong"]
ACTION_NAMES = {0: "Fold", 1: "Call", 2: "Raise"}
ROUND_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_csv(name):
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def add_derived_columns(df):
    """Add p_max, predicted_class columns for BNN data."""
    if "P_weak" in df.columns:
        probs = df[["P_weak", "P_mid", "P_strong"]].values
        df["p_max"] = probs.max(axis=1)
        df["predicted_class"] = probs.argmax(axis=1)
    return df


# =========================================================================
# Priority 1: Confusion Matrix
# =========================================================================

def plot_confusion_matrix():
    print("[1/7] Confusion Matrix...")
    df = load_csv("l1_vs_agg")
    df = add_derived_columns(df)

    # Filter: only showdown hands (true_strength >= 0)
    mask = df["true_strength"] >= 0
    df_show = df[mask].copy()

    y_true = df_show["true_strength"].astype(int)
    y_pred = df_show["predicted_class"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                cbar_kws={"label": "Row fraction"})
    ax.set_xlabel("BNN Predicted Class")
    ax.set_ylabel("True Opponent Strength")
    ax.set_title(f"BNN Confusion Matrix (L1 vs Aggressive)\nAccuracy = {acc:.3f}, N = {len(df_show)}")

    # Add diagonal highlight text
    for i in range(3):
        pct = cm_norm[i, i] * 100
        ax.text(i + 0.5, i + 0.72, f"{pct:.1f}%",
                ha="center", va="center", fontsize=8, color="darkblue",
                fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")
    print(f"  Stats: N={len(df_show)}, Accuracy={acc:.4f}")
    for i, name in enumerate(LABELS):
        row_sum = cm[i].sum()
        if row_sum > 0:
            print(f"    {name}: {cm[i,i]}/{row_sum} = {cm[i,i]/row_sum:.3f}")
    return acc


# =========================================================================
# Priority 2: Calibration Curve
# =========================================================================

def plot_calibration_curve():
    print("[2/7] Calibration Curve...")
    df = load_csv("l1_vs_agg")
    df = add_derived_columns(df)

    mask = df["true_strength"] >= 0
    df_show = df[mask].copy()

    # Use wider bins (0.1) for stability
    bin_width = 0.1
    bins = np.arange(0.3, 1.01, bin_width)
    df_show["conf_bin"] = pd.cut(df_show["p_max"], bins=bins, include_lowest=True)

    bin_stats = df_show.groupby("conf_bin", observed=True).agg(
        count=("p_max", "size"),
        correct=("predicted_class", lambda x: (x == df_show.loc[x.index, "true_strength"].astype(int)).sum()),
    ).reset_index()
    bin_stats["accuracy"] = bin_stats["correct"] / bin_stats["count"]
    bin_stats["p_max_mid"] = bin_stats["conf_bin"].apply(lambda b: b.mid)

    valid = bin_stats["count"] >= 50

    fig, ax = plt.subplots(figsize=(6, 4.5))

    valid_data = bin_stats[valid].sort_values("p_max_mid")
    valid_data["p_max_mid"] = valid_data["p_max_mid"].astype(float)

    # Highlight regions: peak accuracy zone vs overconfidence zone
    ax.axvspan(0.5, 0.7, alpha=0.08, color="green", label="BNN peak accuracy zone")
    ax.axvspan(0.8, 1.0, alpha=0.08, color="red", label="BNN overconfidence zone")

    # Main line: binned accuracy
    ax.plot(valid_data["p_max_mid"], valid_data["accuracy"],
            "o-", color="#1f77b4", markersize=9, linewidth=2.5, label="BNN Accuracy", zorder=5)

    # Annotate peak and overconfidence
    peak_row = valid_data.loc[valid_data["accuracy"].idxmax()]
    ax.annotate(f"Peak: {peak_row['accuracy']:.0%}",
                (peak_row["p_max_mid"], peak_row["accuracy"]),
                textcoords="offset points", xytext=(10, 12),
                ha="left", fontsize=8, color="green", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))

    # Find the highest-confidence point
    high_conf = valid_data[valid_data["p_max_mid"] > 0.8]
    if len(high_conf) > 0:
        hc_row = high_conf.iloc[-1]
        ax.annotate(f"Overconfident: {hc_row['accuracy']:.0%}",
                    (hc_row["p_max_mid"], hc_row["accuracy"]),
                    textcoords="offset points", xytext=(10, -14),
                    ha="left", fontsize=8, color="red", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

    # Random baseline
    ax.axhline(y=0.333, color="gray", linestyle="--", linewidth=1.5, label="Random (0.333)", zorder=3)

    # Perfect calibration line
    ax.plot([0.33, 1.0], [0.33, 1.0], "k-", linewidth=1, alpha=0.3, label="Perfect calibration", zorder=2)

    ax.set_xlabel("Max Prediction Probability (p_max)")
    ax.set_ylabel("Accuracy")
    ax.set_title("BNN Calibration Curve\n(Accuracy peaks at mid-confidence, drops at high confidence)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)
    ax.set_xlim(0.28, 1.02)
    ax.set_ylim(0.15, 0.85)
    ax.grid(axis="y", alpha=0.3)

    for _, row in valid_data.iterrows():
        ax.annotate(f"n={int(row['count'])}",
                    (row["p_max_mid"], row["accuracy"]),
                    textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    out = FIG_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")
    print(f"  Stats: {valid.sum()} bins, total showdown samples={len(df_show)}")
    for _, row in valid_data.iterrows():
        print(f"    p_max~{row['p_max_mid']:.2f}: acc={row['accuracy']:.3f}, n={int(row['count'])}")


# =========================================================================
# Priority 3: L0 vs L1 Bluff/Trap AvgR Comparison
# =========================================================================

def plot_bluff_trap_avgr():
    print("[3/7] L0 vs L1 Bluff/Trap AvgR...")
    df_l0 = load_csv("l0_vs_agg")
    df_l1 = load_csv("l1_vs_agg")

    # Per-hand reward (take first row per hand)
    def per_hand_stats(df, label):
        hand_df = df.groupby("hand_id").agg(
            reward=("reward", "first"),
            hand_type=("hand_type", "first"),
        ).reset_index()
        hand_df["agent"] = label
        return hand_df

    h0 = per_hand_stats(df_l0, "L0 (SARSA)")
    h1 = per_hand_stats(df_l1, "L1 (Belief SARSA)")
    combined = pd.concat([h0, h1], ignore_index=True)

    type_labels = {0: "Normal", 1: "Bluff", 2: "Trap"}
    combined["hand_type_name"] = combined["hand_type"].map(type_labels)

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    agents = ["L0 (SARSA)", "L1 (Belief SARSA)"]
    types = ["Normal", "Bluff", "Trap"]
    x = np.arange(len(types))
    width = 0.35

    for i, agent in enumerate(agents):
        means = []
        sems = []
        for t in types:
            subset = combined[(combined["agent"] == agent) &
                              (combined["hand_type_name"] == t)]["reward"]
            means.append(subset.mean())
            sems.append(subset.sem())
        bars = ax.bar(x + i * width - width / 2, means, width,
                      yerr=sems, capsize=3, label=agent,
                      color=["#d62728", "#1f77b4"][i], alpha=0.85)

    ax.set_xlabel("Hand Type")
    ax.set_ylabel("AvgR (per-hand chip delta)")
    ax.set_title("L0 vs L1: AvgR by Hand Type (vs Aggressive)")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Add value labels manually
    for i, agent in enumerate(agents):
        for j, t in enumerate(types):
            subset = combined[(combined["agent"] == agent) &
                              (combined["hand_type_name"] == t)]["reward"]
            val = subset.mean()
            xpos = j + i * width - width / 2
            ax.annotate(f"{val:+.1f}", xy=(xpos, val),
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "l0_l1_bluff_trap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")
    for agent in agents:
        for t in types:
            subset = combined[(combined["agent"] == agent) &
                              (combined["hand_type_name"] == t)]["reward"]
            print(f"    {agent} vs {t}: AvgR={subset.mean():+.2f}, n={len(subset)}")


# =========================================================================
# Priority 4: Belief Trajectory (Typical Hands)
# =========================================================================

def plot_belief_trajectory():
    print("[4/7] Belief Trajectory...")
    df = load_csv("l1_vs_agg")
    df = add_derived_columns(df)

    # Filter showdown hands with complete round coverage
    mask = df["true_strength"] >= 0
    df_show = df[mask].copy()

    # Find typical hands: one per true_strength class
    # Pick hands with most decision points across rounds
    selected_hands = []
    for ts_class in [0, 1, 2]:
        class_hands = df_show[df_show["true_strength"] == ts_class]
        # Pick hand with most decision points
        hand_counts = class_hands.groupby("hand_id").size()
        if len(hand_counts) > 0:
            best_hand = hand_counts.idxmax()
            selected_hands.append((best_hand, ts_class))

    # Also find a "belief update" hand: where predicted changes across rounds
    update_candidates = []
    for hid, group in df_show.groupby("hand_id"):
        if len(group) >= 3:
            preds = group["predicted_class"].values
            if len(set(preds)) >= 2:  # prediction changes
                update_candidates.append(hid)
    if update_candidates:
        selected_hands.append((update_candidates[0], "update"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    colors = {"Weak": "#d62728", "Mid": "#ff7f0e", "Strong": "#2ca02c"}

    for idx, (hid, label) in enumerate(selected_hands[:4]):
        ax = axes[idx]
        hand_df = df_show[df_show["hand_id"] == hid].sort_values("step")

        if label == "update":
            title = f"Hand #{hid} (Belief Update)"
        else:
            title = f"Hand #{hid} (True: {LABELS[label]})"

        rounds = hand_df["round"].values
        p_weak = hand_df["P_weak"].values
        p_mid = hand_df["P_mid"].values
        p_strong = hand_df["P_strong"].values

        # Stacked area chart
        ax.fill_between(range(len(rounds)), 0, p_weak,
                        alpha=0.7, color=colors["Weak"], label="P(Weak)")
        ax.fill_between(range(len(rounds)), p_weak, p_weak + p_mid,
                        alpha=0.7, color=colors["Mid"], label="P(Mid)")
        ax.fill_between(range(len(rounds)), p_weak + p_mid, 1.0,
                        alpha=0.7, color=colors["Strong"], label="P(Strong)")

        # Round labels on x-axis
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([ROUND_NAMES.get(int(r), f"R{r}") for r in rounds],
                           rotation=30, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.legend(loc="upper right", fontsize=7)

        # Add equity annotation
        equities = hand_df["equity"].values
        ax.annotate(f"equity={equities[-1]:.2f}",
                    xy=(len(rounds) - 1, 0.02), fontsize=7, color="gray")

    fig.suptitle("Belief Trajectories: BNN Predictions Across Betting Rounds",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "belief_trajectory.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")
    print(f"  Selected hands: {[(h, l) for h, l in selected_hands[:4]]}")


# =========================================================================
# Priority 5: L3 Gate Heatmap
# =========================================================================

def plot_l3_gate_heatmap():
    print("[5/7] L3 Gate Heatmap...")
    df = load_csv("l3_vs_agg")
    df = add_derived_columns(df)

    equity = df["equity"].values
    p_max = df["p_max"].values
    corr = df["correction_magnitude"].values

    # Discretize into 10 bins each
    eq_bins = np.clip(np.digitize(equity, np.linspace(0, 1, 11)) - 1, 0, 9)
    pm_bins = np.clip(np.digitize(p_max, np.linspace(0.33, 1.0, 11)) - 1, 0, 9)

    heatmap = np.full((10, 10), np.nan)
    counts = np.zeros((10, 10), dtype=int)

    for eq_b in range(10):
        for pm_b in range(10):
            mask = (eq_bins == eq_b) & (pm_bins == pm_b)
            if mask.sum() > 0:
                heatmap[pm_b, eq_b] = corr[mask].mean()
                counts[pm_b, eq_b] = mask.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    # Mask NaN cells
    masked_heatmap = np.ma.masked_invalid(heatmap)
    sns.heatmap(masked_heatmap, ax=ax, cmap="YlOrRd",
                xticklabels=[f"{i*0.1:.1f}" for i in range(10)],
                yticklabels=[f"{0.33+i*0.067:.2f}" for i in range(10)],
                cbar_kws={"label": "Mean |gate_delta| (L2 norm)"})
    ax.set_xlabel("Equity (phi)")
    ax.set_ylabel("p_max (BNN Confidence)")
    ax.set_title("L3 Gate Correction Magnitude: Equity x BNN Confidence")

    # Annotate counts
    for i in range(10):
        for j in range(10):
            if counts[i, j] > 0 and not np.isnan(heatmap[i, j]):
                ax.text(j + 0.5, i + 0.5, f"n={counts[i,j]}",
                        ha="center", va="center", fontsize=6, color="darkblue")

    plt.tight_layout()
    out = FIG_DIR / "l3_gate_heatmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")
    print(f"  Stats: mean_corr={np.nanmean(heatmap):.2f}, "
          f"max_cell={np.nanmax(heatmap):.2f}, min_cell={np.nanmin(heatmap):.2f}")


# =========================================================================
# Priority 6: Correction vs Confidence Scatter
# =========================================================================

def plot_correction_vs_confidence():
    print("[6/7] Correction vs Confidence Scatter...")
    df_l2 = load_csv("l2_vs_agg")
    df_l3 = load_csv("l3_vs_agg")
    df_l2 = add_derived_columns(df_l2)
    df_l3 = add_derived_columns(df_l3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, df, label, color in [(axes[0], df_l2, "L2 (Tabular+Gate)", "#d62728"),
                                  (axes[1], df_l3, "L3 (Neural+Gate)", "#1f77b4")]:
        p_max = df["p_max"].values
        corr = df["correction_magnitude"].values

        # Clip extreme values for visualization (99th percentile)
        clip_val = np.percentile(corr, 99)
        corr_clipped = np.clip(corr, 0, clip_val * 1.2)

        ax.scatter(p_max, corr_clipped, alpha=0.15, s=3, color=color, rasterized=True)
        ax.axvline(x=0.65, color="black", linestyle="--", linewidth=1.5,
                    label=r"$\tau=0.65$")

        # Add trend line (lowess-like: bin and plot mean)
        bins = np.linspace(0.33, 1.0, 20)
        bin_idx = np.digitize(p_max, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
        bin_means = [corr[bin_idx == i].mean() for i in range(len(bins) - 1)]
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
        valid = [i for i in range(len(bin_means)) if not np.isnan(bin_means[i])]
        if valid:
            ax.plot([bin_centers[i] for i in valid],
                    [bin_means[i] for i in valid],
                    "-", color="orange", linewidth=2, label="Mean trend")

        ax.set_xlabel("p_max (BNN Confidence)")
        ax.set_title(label)
        ax.legend(fontsize=8)

        # Stats
        low_conf = corr[p_max < 0.65]
        high_conf = corr[p_max >= 0.65]
        print(f"  {label}: p_max<0.65: mean_corr={low_conf.mean():.2f}, "
              f"n={len(low_conf)} | p_max>=0.65: mean_corr={high_conf.mean():.2f}, "
              f"n={len(high_conf)}")

    axes[0].set_ylabel("Gate Correction Magnitude ||delta||_2")
    fig.suptitle("Gate Correction vs BNN Confidence", fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "correction_vs_confidence.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")


# =========================================================================
# Priority 7: Correction Direction Analysis
# =========================================================================

def plot_correction_direction():
    print("[7/7] Correction Direction Analysis...")
    df_l2 = load_csv("l2_vs_agg")
    df_l3 = load_csv("l3_vs_agg")
    df_l2 = add_derived_columns(df_l2)
    df_l3 = add_derived_columns(df_l3)

    # Combine L2 and L3 data
    combined = pd.concat([
        df_l2[["true_strength", "gate_delta_0", "gate_delta_1", "gate_delta_2", "p_max"]].assign(agent="L2"),
        df_l3[["true_strength", "gate_delta_0", "gate_delta_1", "gate_delta_2", "p_max"]].assign(agent="L3"),
    ], ignore_index=True)

    # Filter: only showdown hands
    combined = combined[combined["true_strength"] >= 0].copy()
    combined["true_strength"] = combined["true_strength"].astype(int)

    # Determine correction direction: which action's logit increased most
    deltas = combined[["gate_delta_0", "gate_delta_1", "gate_delta_2"]].values
    max_delta_idx = deltas.argmax(axis=1)  # 0=Fold, 1=Call, 2=Raise

    # Classify correction direction
    direction_labels = {0: "More Fold", 1: "More Call", 2: "More Raise"}
    combined["correction_dir"] = [direction_labels[i] for i in max_delta_idx]

    # Only consider cases where gate actually does something (non-trivial delta)
    delta_norms = np.linalg.norm(deltas, axis=1)
    combined["delta_norm"] = delta_norms
    # Filter: gate has meaningful effect (above median)
    threshold = np.percentile(delta_norms, 50)
    meaningful = combined[delta_norms >= threshold].copy()

    # Stacked bar chart: true_strength x correction_direction
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax_idx, (agent_name, agent_data) in enumerate(
            [("L2", combined[combined["agent"] == "L2"]),
             ("L3", combined[combined["agent"] == "L3"])]):
        ax = axes[ax_idx]
        directions = ["More Fold", "More Call", "More Raise"]
        true_classes = [0, 1, 2]

        bottom = np.zeros(3)
        colors_dir = {"More Fold": "#d62728", "More Call": "#2ca02c", "More Raise": "#1f77b4"}

        for dir_name in directions:
            counts = []
            for tc in true_classes:
                subset = agent_data[(agent_data["true_strength"] == tc) &
                                    (agent_data["correction_dir"] == dir_name)]
                total_tc = (agent_data["true_strength"] == tc).sum()
                counts.append(len(subset) / total_tc * 100 if total_tc > 0 else 0)
            ax.bar(range(3), counts, bottom=bottom, label=dir_name,
                   color=colors_dir[dir_name], alpha=0.85)
            bottom += np.array(counts)

        ax.set_xticks(range(3))
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("True Opponent Strength")
        ax.set_ylabel("Fraction (%)")
        ax.set_title(f"{agent_name}: Gate Correction Direction")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 100)

    fig.suptitle("Gate Correction Direction by True Opponent Strength",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "correction_direction.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")

    # Stats
    for agent_name in ["L2", "L3"]:
        ad = combined[combined["agent"] == agent_name]
        print(f"  {agent_name} (N={len(ad)}):")
        for tc, name in enumerate(LABELS):
            subset = ad[ad["true_strength"] == tc]
            if len(subset) > 0:
                dirs = subset["correction_dir"].value_counts(normalize=True)
                print(f"    {name}: {', '.join(f'{k}={v:.1%}' for k, v in dirs.items())}")


# =========================================================================
# Priority 8: Gate Effect on Decision Quality (BNN Confidence vs AvgR)
# =========================================================================

def plot_gate_effect():
    """2-panel: (a) BNN accuracy vs Gate intervention (complementary), (b) L1 vs L2 AvgR."""
    print("[8/8] Gate Effect: BNN Accuracy vs Gate Intervention + L1/L2 AvgR...")
    df_l1 = load_csv("l1_vs_agg")
    df_l2 = load_csv("l2_vs_agg")
    df_l1 = add_derived_columns(df_l1)
    df_l2 = add_derived_columns(df_l2)

    # Use 0.1-width bins consistent with calibration curve
    bin_width = 0.1
    bins = np.arange(0.3, 1.01, bin_width)

    # ========== Panel (a): BNN Accuracy vs Gate Intervention Rate ==========
    # This shows the COMPLEMENTARY relationship:
    # BNN accuracy peaks at mid-confidence → gate intervenes LEAST there
    # BNN accuracy drops at high-confidence → gate intervenes MOST there
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]

    # BNN accuracy by confidence (from L1 data, same as calibration)
    mask = df_l1["true_strength"] >= 0
    df_cal = df_l1[mask].copy()
    df_cal["conf_bin"] = pd.cut(df_cal["p_max"], bins=bins, include_lowest=True)
    cal_stats = df_cal.groupby("conf_bin", observed=True).agg(
        count=("p_max", "size"),
        correct=("predicted_class", lambda x: (x == df_cal.loc[x.index, "true_strength"].astype(int)).sum()),
    ).reset_index()
    cal_stats["accuracy"] = cal_stats["correct"] / cal_stats["count"]
    cal_stats["conf_mid"] = cal_stats["conf_bin"].apply(lambda b: b.mid)
    cal_valid = cal_stats[cal_stats["count"] >= 50].sort_values("conf_mid")

    # Gate intervention rate by confidence (from L2 data)
    df2 = df_l2.copy()
    df2["conf_bin"] = pd.cut(df2["p_max"], bins=bins, include_lowest=True)
    gate_stats = df2.groupby("conf_bin", observed=True).agg(
        n=("correction_magnitude", "size"),
        change_count=("original_action", lambda x: (x != df2.loc[x.index, "final_action"]).sum()),
    ).reset_index()
    gate_stats["change_rate"] = gate_stats["change_count"] / gate_stats["n"] * 100
    gate_stats["conf_mid"] = gate_stats["conf_bin"].apply(lambda b: b.mid)
    gate_valid = gate_stats[gate_stats["n"] >= 50].sort_values("conf_mid")

    # Plot BNN accuracy (left y-axis)
    ax.plot(cal_valid["conf_mid"], cal_valid["accuracy"] * 100,
            "o-", color="#1f77b4", markersize=9, linewidth=2.5,
            label="BNN Accuracy", zorder=5)
    for _, row in cal_valid.iterrows():
        ax.annotate(f"{row['accuracy']*100:.0f}%", (row["conf_mid"], row["accuracy"] * 100),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color="#1f77b4", fontweight="bold")

    # Plot gate intervention rate (right y-axis)
    ax2 = ax.twinx()
    ax2.plot(gate_valid["conf_mid"], gate_valid["change_rate"],
             "s--", color="#d62728", markersize=8, linewidth=2.5,
             label="Gate Intervention Rate", zorder=5)
    for _, row in gate_valid.iterrows():
        ax2.annotate(f"{row['change_rate']:.0f}%", (row["conf_mid"], row["change_rate"]),
                     textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=7.5, color="#d62728", fontweight="bold")

    ax.set_xlabel("BNN Confidence (p_max)")
    ax.set_ylabel("BNN Accuracy (%)", color="#1f77b4")
    ax2.set_ylabel("Gate Intervention Rate (%)", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("(a) BNN Accuracy vs Gate Intervention (Complementary)")
    ax.set_xticks([0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    ax.grid(axis="x", alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9, fontsize=8)

    # Annotate sample sizes
    for _, row in cal_valid.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row["conf_mid"], 18),
                    ha="center", fontsize=6, color="gray")

    # ========== Panel (b): L1 vs L2 AvgR by confidence ==========
    ax3 = axes[1]

    def _hand_avgr_by_conf(df):
        hand_first = df.groupby("hand_id").agg(
            p_max_first=("p_max", "first"),
            reward=("reward", "first"),
        ).reset_index()
        hand_first["conf_bin"] = pd.cut(hand_first["p_max_first"], bins=bins, include_lowest=True)
        binned = hand_first.groupby("conf_bin", observed=True).agg(
            avgr=("reward", "mean"),
            sem=("reward", "sem"),
            count=("reward", "size"),
            p_max_mid=("p_max_first", "mean"),
        ).reset_index()
        return binned[binned["count"] >= 30].sort_values("p_max_mid")

    l1_stats = _hand_avgr_by_conf(df_l1)
    l2_stats = _hand_avgr_by_conf(df_l2)

    ax3.plot(l1_stats["p_max_mid"], l1_stats["avgr"],
             "o-", color="#d62728", markersize=8, linewidth=2.5,
             label="L1 (no gate)", zorder=5)
    ax3.fill_between(l1_stats["p_max_mid"],
                     l1_stats["avgr"] - l1_stats["sem"],
                     l1_stats["avgr"] + l1_stats["sem"],
                     alpha=0.15, color="#d62728")
    ax3.plot(l2_stats["p_max_mid"], l2_stats["avgr"],
             "s-", color="#1f77b4", markersize=8, linewidth=2.5,
             label="L2 (+gate)", zorder=5)
    ax3.fill_between(l2_stats["p_max_mid"],
                     l2_stats["avgr"] - l2_stats["sem"],
                     l2_stats["avgr"] + l2_stats["sem"],
                     alpha=0.15, color="#1f77b4")
    ax3.axhline(y=0, color="black", linewidth=0.8, zorder=3)
    ax3.set_xlabel("BNN Confidence (p_max)")
    ax3.set_ylabel("AvgR (per-hand)")
    ax3.set_title("(b) L1 vs L2: Gate Benefit by Confidence")
    ax3.legend(loc="lower left", framealpha=0.9)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_xticks([0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    for _, row in l1_stats.iterrows():
        ax3.annotate(f"{row['avgr']:+.0f}\nn={int(row['count'])}",
                     (row["p_max_mid"], row["avgr"]),
                     textcoords="offset points", xytext=(-20, 6),
                     ha="center", fontsize=6.5, color="#d62728")
    for _, row in l2_stats.iterrows():
        ax3.annotate(f"{row['avgr']:+.0f}\nn={int(row['count'])}",
                     (row["p_max_mid"], row["avgr"]),
                     textcoords="offset points", xytext=(20, 6),
                     ha="center", fontsize=6.5, color="#1f77b4")

    fig.suptitle("Gate Effect Analysis (L1 vs L2 Ablation)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "gate_effect.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # Stats
    l1_overall = df_l1.groupby("hand_id")["reward"].first().mean()
    l2_overall = df_l2.groupby("hand_id")["reward"].first().mean()
    print(f"  Overall AvgR: L1={l1_overall:+.2f}, L2={l2_overall:+.2f}, delta={l2_overall-l1_overall:+.2f}")
    print(f"  BNN Accuracy vs Gate Intervention (complementary):")
    for _, row in cal_valid.iterrows():
        gate_row = gate_valid[gate_valid["conf_mid"].apply(lambda x: abs(x - row["conf_mid"]) < 0.01)]
        gr = gate_row["change_rate"].values[0] if len(gate_row) > 0 else None
        gr_str = f"{gr:.1f}%" if gr is not None else "N/A"
        print(f"    p_max~{row['conf_mid']:.2f}: BNN_acc={row['accuracy']*100:.1f}%, "
              f"gate_interv={gr_str}, n_cal={int(row['count'])}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("CS181 Visualization Experiments")
    print("=" * 60)
    print(f"Data dir: {DATA_DIR}")
    print(f"Figure dir: {FIG_DIR}\n")

    # Check data exists
    for name in ["l1_vs_agg", "l3_vs_agg", "l2_vs_agg", "l0_vs_agg"]:
        p = DATA_DIR / f"{name}.csv"
        if not p.exists():
            print(f"ERROR: {p} not found! Run collect_viz_data.py first.")
            return

    acc = plot_confusion_matrix()
    print()
    plot_calibration_curve()
    print()
    plot_bluff_trap_avgr()
    print()
    plot_belief_trajectory()
    print()
    plot_l3_gate_heatmap()
    print()
    plot_correction_vs_confidence()
    print()
    plot_correction_direction()
    print()
    plot_gate_effect()

    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)

    # List all generated figures
    for f in sorted(FIG_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()

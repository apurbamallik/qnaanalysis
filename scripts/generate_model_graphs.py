"""
EduBench-Local — Model Evaluation Graph Generator
===================================================
Generates publication-quality figures matching benchmark specifications:
  1. Metric Correlation Analysis (3-panel: ROUGE-L vs BERTScore, ROUGE-L vs Judge, BERTScore vs Judge)
  2. Generation Latency by Dataset (Horizontal bar chart with error bars and data labels)
  3. Score Heatmap (Dataset x Metric: ROUGE-L, BERTScore F1, Judge Score)
  4. Score Distributions by Dataset (3-panel boxplots: ROUGE-L, BERTScore, Judge Score)
  5. Multi-Metric Profile by Dataset (3-axis Radar Chart: ROUGE-L, BERTScore, Judge)
  6. Quality vs. Latency Trade-off / Efficiency Frontier
  7. Dataset Quality Breakdown (LLM-as-Judge 1-5 Bar Chart)
  8. Output Length / Token Count by Dataset

Generates individual suites for both benchmarked models:
  - Llama 3.2 (3B) [llama3.2:3b]
  - DeepSeek-R1 (1.5B) [deepseek-r1:1.5b]
Plus a dedicated cross-model comparative suite.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Consistent visual palette
DATASET_COLORS = {
    "SciQ": "#E07A8B",          # Soft coral / rose
    "OpenBookQA": "#C8B18B",    # Warm beige / golden tan
    "ARC-Challenge": "#5DB4C3", # Teal / cyan
    "ARC-Easy": "#76C68F",      # Fresh green
    "RACE": "#4E79A7",          # Steel blue
    "SQuAD": "#C572BE",         # Soft magenta / violet
    "General": "#94A3B8"
}

MODEL_CONFIGS = {
    "llama3.2:3b": {
        "display_name": "Llama 3.2 3B",
        "short_folder": "llama_3.2_3b"
    },
    "deepseek-r1:1.5b": {
        "display_name": "DeepSeek-R1 1.5B",
        "short_folder": "deepseek_r1_1.5b"
    }
}

def format_p(p_val: float) -> str:
    """Format p-value nicely for publication plots."""
    if p_val == 0.0 or p_val < 1e-100:
        return f"{p_val:.1e}"
    elif p_val < 0.001:
        return f"{p_val:.1e}"
    else:
        return f"{p_val:.3f}"

def set_plot_style():
    """Set publication-grade aesthetic defaults."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "axes.edgecolor": "#94A3B8",
        "axes.linewidth": 0.8,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF"
    })

# ---------------------------------------------------------------------------
# GRAPH 1: Metric Correlation Analysis (3 subplots)
# ---------------------------------------------------------------------------
def plot_metric_correlation(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    if len(mdf) < 5:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"Metric Correlation Analysis \u2014 {display_name}", fontsize=14, fontweight="bold", y=1.02)

    pairs = [
        ("rouge_l", "bertscore_f1", "ROUGE-L", "BERTScore F1"),
        ("rouge_l", "judge_score", "ROUGE-L", "Judge Score"),
        ("bertscore_f1", "judge_score", "BERTScore F1", "Judge Score")
    ]

    datasets = sorted(mdf["source_dataset"].unique())

    for idx, (x_col, y_col, x_label, y_label) in enumerate(pairs):
        ax = axes[idx]
        x_vals = mdf[x_col].values
        y_vals = mdf[y_col].values

        # Correlation
        rho, p_val = spearmanr(x_vals, y_vals)
        p_str = format_p(p_val)
        ax.set_title(f"{x_label} vs {y_label}\n(\u03c1={rho:.3f}, p={p_str})", fontweight="bold", fontsize=11, pad=8)

        # Scatter by dataset
        for ds in datasets:
            sub = mdf[mdf["source_dataset"] == ds]
            color = DATASET_COLORS.get(ds, "#3B82F6")
            ax.scatter(sub[x_col], sub[y_col], color=color, label=ds, alpha=0.55, s=32, edgecolors="none")

        # Regression line
        if len(x_vals) > 2 and np.std(x_vals) > 1e-6:
            poly = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            y_line = poly[0] * x_line + poly[1]
            ax.plot(x_line, y_line, color="gray", linestyle="--", linewidth=1.2, alpha=0.85)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle="-", alpha=0.35, color="#CBD5E1")

        if idx == 0:
            ax.legend(title="dataset", loc="lower right", fontsize=8.5, title_fontsize=9, framealpha=0.9, edgecolor="#E2E8F0")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "01_metric_correlation_analysis.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# GRAPH 2: Generation Latency by Dataset (Horizontal Bar Chart)
# ---------------------------------------------------------------------------
def plot_generation_latency(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    stats = mdf.groupby("source_dataset")["latency_sec"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0)
    stats = stats.sort_values("mean", ascending=True).reset_index(drop=True)  # ascending so top bar has highest latency in barh

    fig, ax = plt.subplots(figsize=(10, max(4.2, len(stats) * 0.75)))
    y_pos = np.arange(len(stats))

    # Reverse viridis so top (highest latency) gets vibrant lime/green
    cmap = plt.cm.viridis
    norm_vals = np.linspace(0.2, 0.85, len(stats))
    colors = [cmap(v) for v in norm_vals]

    bars = ax.barh(y_pos, stats["mean"], xerr=stats["std"], color=colors, height=0.68,
                   capsize=4.5, edgecolor="none",
                   error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2, "alpha": 0.9})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats["source_dataset"], fontsize=11, fontweight="medium")
    ax.set_xlabel("Average Latency (seconds)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(f"Generation Latency by Dataset \u2014 {display_name}", fontsize=13, fontweight="bold", pad=14)

    max_x = max(stats["mean"] + stats["std"]) * 1.15
    ax.set_xlim(0, max_x)

    for i, bar in enumerate(bars):
        mean_val = stats["mean"].iloc[i]
        std_val = stats["std"].iloc[i]
        x_text = mean_val + std_val + (max_x * 0.015)
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{mean_val:.1f}s",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#1E293B")

    ax.xaxis.grid(True, linestyle="-", alpha=0.35, color="#E2E8F0")
    ax.yaxis.grid(False)
    plt.tight_layout()

    out_file = os.path.join(out_dir, "02_generation_latency_by_dataset.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# GRAPH 3: Score Heatmap (Dataset x Metric)
# ---------------------------------------------------------------------------
def plot_score_heatmap(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    table = mdf.groupby("source_dataset")[["rouge_l", "bertscore_f1", "judge_score"]].mean()
    table = table.rename(columns={
        "rouge_l": "ROUGE-L",
        "bertscore_f1": "BERTScore F1",
        "judge_score": "Judge Score (1-5)"
    })
    table = table.sort_index()

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(table) * 0.85)))
    sns.heatmap(
        table,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Score"},
        linewidths=1.5,
        linecolor="white",
        annot_kws={"fontsize": 11, "fontweight": "medium"},
        ax=ax
    )

    ax.set_title(f"{display_name} \u2014 Score Heatmap (Dataset \u00d7 Metric)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Metric", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Dataset", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()

    out_file = os.path.join(out_dir, "03_score_heatmap.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# GRAPH 4: Score Distributions by Dataset (3-Panel Boxplots)
# ---------------------------------------------------------------------------
def plot_score_distributions(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Score Distributions by Dataset \u2014 {display_name}", fontsize=14, fontweight="bold", y=1.02)

    configs = [
        ("rouge_l", "ROUGE-L F1", False),
        ("bertscore_f1", "BERTScore F1", False),
        ("judge_score", "Judge Score (1-5)", True)
    ]

    for idx, (col, title, is_judge) in enumerate(configs):
        ax = axes[idx]
        # Sort datasets by mean descending for that metric
        order = mdf.groupby("source_dataset")[col].median().sort_values(ascending=False).index.tolist()
        palette = [DATASET_COLORS.get(ds, "#3B82F6") for ds in order]

        sns.boxplot(
            data=mdf,
            x="source_dataset",
            y=col,
            order=order,
            hue="source_dataset",
            palette=palette,
            legend=False,
            ax=ax,
            linewidth=1.1,
            flierprops=dict(marker="o", markersize=4.5, markerfacecolor="none", markeredgecolor="#475569", alpha=0.7),
            boxprops=dict(alpha=0.85)
        )

        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.yaxis.grid(True, linestyle="-", alpha=0.35, color="#E2E8F0")

        if is_judge:
            ax.set_ylim(0.8, 5.2)
            ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    plt.tight_layout()
    out_file = os.path.join(out_dir, "04_score_distributions_by_dataset.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# GRAPH 5: Multi-Metric Profile by Dataset (Radar Chart)
# ---------------------------------------------------------------------------
def plot_multi_metric_radar(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    metrics = ["rouge_l", "bertscore_f1", "judge_score"]
    categories = ["ROUGE-L", "BERTScore", "Judge"]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    plot_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    datasets = sorted(mdf["source_dataset"].unique())
    for ds in datasets:
        sub = mdf[mdf["source_dataset"] == ds]
        avg_rouge = sub["rouge_l"].mean()
        avg_bert = sub["bertscore_f1"].mean()
        norm_judge = sub["judge_score"].mean() / 5.0  # Normalized 1-5 to 0-1 scale

        values = [avg_rouge, avg_bert, norm_judge]
        values += values[:1]

        color = DATASET_COLORS.get(ds, "#3B82F6")
        ax.plot(plot_angles, values, color=color, linewidth=2.2, marker="o", markersize=6, label=ds)
        ax.fill(plot_angles, values, color=color, alpha=0.15)

    ax.set_theta_offset(0)
    ax.set_theta_direction(1)
    ax.set_thetagrids(np.degrees(angles), categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9.5, color="#64748B")

    ax.grid(True, linestyle="-", color="#CBD5E1", alpha=0.7)
    ax.set_title(f"{display_name} \u2014 Multi-Metric Profile by Dataset", fontsize=13, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.08), fontsize=10, framealpha=0.9, edgecolor="#E2E8F0")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "05_multi_metric_profile_radar.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# GRAPH 6: Efficiency Frontier (Quality vs. Generation Latency Trade-off)
# ---------------------------------------------------------------------------
def plot_efficiency_frontier(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    stats = mdf.groupby("source_dataset").agg(
        judge_score=("judge_score", "mean"),
        latency_sec=("latency_sec", "mean"),
        count=("id", "count")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for _, row in stats.iterrows():
        ds = row["source_dataset"]
        color = DATASET_COLORS.get(ds, "#3B82F6")
        ax.scatter(row["latency_sec"], row["judge_score"], color=color, s=260,
                   edgecolor="#1E293B", linewidth=1.5, zorder=5, label=ds)
        ax.annotate(
            f"{ds}\n({row['judge_score']:.2f} pts, {row['latency_sec']:.2f}s)",
            (row["latency_sec"], row["judge_score"]),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#1E293B"
        )

    ax.axhline(y=4.0, color="#10B981", linestyle="--", linewidth=1.4, alpha=0.8, label="Proficiency Threshold (4.0/5.0)")

    min_lat, max_lat = stats["latency_sec"].min(), stats["latency_sec"].max()
    lat_pad = max(0.1, (max_lat - min_lat) * 0.35)
    ax.set_xlim(max(0, min_lat - lat_pad), max_lat + lat_pad * 1.5)

    min_j, max_j = stats["judge_score"].min(), stats["judge_score"].max()
    ax.set_ylim(max(1.0, min_j - 0.4), min(5.2, max_j + 0.4))

    ax.set_title(f"{display_name} \u2014 Quality vs. Latency Trade-off by Dataset", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Average Generation Latency (seconds)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Average LLM Judge Score (1\u20135)", fontsize=11, fontweight="bold", labelpad=8)
    ax.grid(True, linestyle="-", alpha=0.35, color="#CBD5E1")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#E2E8F0")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "06_efficiency_frontier.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# GRAPH 7 (Bonus): Dataset Quality Breakdown (LLM Judge Bar Chart)
# ---------------------------------------------------------------------------
def plot_dataset_quality_breakdown(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    stats = mdf.groupby("source_dataset")["judge_score"].agg(["mean", "std"]).reset_index()
    stats = stats.sort_values("mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [DATASET_COLORS.get(ds, "#3B82F6") for ds in stats["source_dataset"]]

    bars = ax.bar(stats["source_dataset"], stats["mean"], yerr=stats["std"], color=colors,
                  width=0.55, edgecolor="none", capsize=4, error_kw={"ecolor": "#334155", "elinewidth": 1.2})

    ax.axhline(y=4.0, color="#10B981", linestyle="--", linewidth=1.4, alpha=0.8, label="Proficiency Threshold (4.0/5.0)")
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("LLM-as-Judge Score (1\u20135)", fontsize=11, fontweight="bold")
    ax.set_title(f"{display_name} \u2014 Pedagogical Quality by Dataset", fontsize=13, fontweight="bold", pad=12)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}/5.0",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.yaxis.grid(True, linestyle="-", alpha=0.35, color="#E2E8F0")
    ax.xaxis.grid(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    out_file = os.path.join(out_dir, "07_dataset_quality_breakdown.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# GRAPH 8 (Bonus): Token Generation Count by Dataset
# ---------------------------------------------------------------------------
def plot_token_count_by_dataset(df: pd.DataFrame, model_name: str, display_name: str, out_dir: str):
    mdf = df[df["model"] == model_name].copy()
    if "token_count" not in mdf.columns or mdf["token_count"].dropna().empty:
        return
    stats = mdf.groupby("source_dataset")["token_count"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0)
    stats = stats.sort_values("mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4.0, len(stats) * 0.75)))
    y_pos = np.arange(len(stats))
    colors = plt.cm.magma(np.linspace(0.4, 0.8, len(stats)))

    bars = ax.barh(y_pos, stats["mean"], xerr=stats["std"], color=colors, height=0.65,
                   capsize=4, edgecolor="none", error_kw={"ecolor": "black", "elinewidth": 1.2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats["source_dataset"], fontsize=11)
    ax.set_xlabel("Average Output Tokens", fontsize=11, fontweight="bold")
    ax.set_title(f"Output Token Length by Dataset \u2014 {display_name}", fontsize=13, fontweight="bold", pad=12)

    max_x = max(stats["mean"] + stats["std"]) * 1.15
    ax.set_xlim(0, max_x)

    for i, bar in enumerate(bars):
        mean_val = stats["mean"].iloc[i]
        std_val = stats["std"].iloc[i]
        x_text = mean_val + std_val + (max_x * 0.015)
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{mean_val:.0f} tokens",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#1E293B")

    ax.xaxis.grid(True, linestyle="-", alpha=0.35, color="#E2E8F0")
    ax.yaxis.grid(False)
    plt.tight_layout()

    out_file = os.path.join(out_dir, "08_token_count_by_dataset.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {out_file}")

# ---------------------------------------------------------------------------
# COMPARATIVE SUITE: Both Models Side-by-Side
# ---------------------------------------------------------------------------
def plot_cross_model_comparison(df: pd.DataFrame, out_dir: str):
    """Generates direct head-to-head comparison figures for both models."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Multi-Metric Profile Comparison (Grouped Bar Chart across common datasets)
    common_datasets = sorted(set(df[df["model"]=="llama3.2:3b"]["source_dataset"]).intersection(
                             set(df[df["model"]=="deepseek-r1:1.5b"]["source_dataset"])))
    
    comp_df = df[df["source_dataset"].isin(common_datasets)].copy()

    # Latency Comparison Bar
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lat_summary = comp_df.groupby(["source_dataset", "model"])["latency_sec"].mean().reset_index()
    lat_summary["model_label"] = lat_summary["model"].map(lambda m: MODEL_CONFIGS.get(m, {}).get("display_name", m))
    
    sns.barplot(data=lat_summary, x="source_dataset", y="latency_sec", hue="model_label",
                palette=["#3B82F6", "#10B981"], ax=ax)
    ax.set_title("Generation Latency Comparison (SciQ & RACE)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Average Latency (seconds)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Benchmark Dataset", fontsize=11, fontweight="bold")
    ax.legend(title="Model", framealpha=0.9)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}s", (p.get_x() + p.get_width() / 2, h + 0.08),
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_latency_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Score Heatmap Comparison (All models & metrics)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    matrix = comp_df.groupby(["model", "source_dataset"])[["rouge_l", "bertscore_f1", "judge_score"]].mean()
    matrix = matrix.rename(columns={"rouge_l": "ROUGE-L", "bertscore_f1": "BERTScore F1", "judge_score": "Judge Score (1-5)"})
    matrix.index = [f"{MODEL_CONFIGS.get(m, {}).get('display_name', m)} \u2014 {ds}" for m, ds in matrix.index]
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=1.2, linecolor="white", ax=ax)
    ax.set_title("Cross-Model Performance Heatmap (Dataset \u00d7 Metric)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_score_heatmap_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Tri-metric Radar Comparison
    categories = ["ROUGE-L", "BERTScore F1", "Judge (Norm)"]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    plot_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    model_colors = {"llama3.2:3b": "#3B82F6", "deepseek-r1:1.5b": "#10B981"}

    for m in ["llama3.2:3b", "deepseek-r1:1.5b"]:
        sub = comp_df[comp_df["model"] == m]
        vals = [sub["rouge_l"].mean(), sub["bertscore_f1"].mean(), sub["judge_score"].mean() / 5.0]
        vals += vals[:1]
        name = MODEL_CONFIGS.get(m, {}).get("display_name", m)
        col = model_colors.get(m, "#6366F1")
        ax.plot(plot_angles, vals, color=col, linewidth=2.5, marker="o", markersize=7, label=name)
        ax.fill(plot_angles, vals, color=col, alpha=0.18)

    ax.set_theta_offset(0)
    ax.set_theta_direction(1)
    ax.set_thetagrids(np.degrees(angles), categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Cross-Model Tri-Metric Radar Comparison (SciQ & RACE)", fontsize=13, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_multi_metric_radar_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Efficiency Frontier Comparison
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    eff_data = df.groupby(["model", "source_dataset"]).agg(
        judge=("judge_score", "mean"),
        latency=("latency_sec", "mean")
    ).reset_index()

    for _, row in eff_data.iterrows():
        m = row["model"]
        ds = row["source_dataset"]
        name = MODEL_CONFIGS.get(m, {}).get("display_name", m)
        col = model_colors.get(m, "#64748B")
        marker = "o" if m == "llama3.2:3b" else "s"
        ax.scatter(row["latency"], row["judge"], color=col, s=220, marker=marker,
                   edgecolor="#0F172A", linewidth=1.2, alpha=0.9, zorder=5)
        ax.annotate(f"{name}\n[{ds}]", (row["latency"], row["judge"]),
                    xytext=(8, -4), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.axhline(y=4.0, color="#10B981", linestyle="--", alpha=0.7, label="Proficiency Threshold (4.0/5.0)")
    ax.set_xlabel("Average Latency (seconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average LLM Judge Score (1\u20135)", fontsize=11, fontweight="bold")
    ax.set_title("Overall Efficiency Frontier: Quality vs. Latency by Model & Dataset", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="-", alpha=0.35, color="#CBD5E1")
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_efficiency_frontier_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved comparative suite to: {out_dir}")

# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------
def main():
    set_plot_style()
    data_path = os.path.join("results", "scored_results.csv")
    if not os.path.exists(data_path):
        print(f"[!] File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"[*] Loaded {len(df)} records from {data_path}.")

    base_out = "model_graphs"
    os.makedirs(base_out, exist_ok=True)

    for model_key, cfg in MODEL_CONFIGS.items():
        display_name = cfg["display_name"]
        model_out = os.path.join(base_out, cfg["short_folder"])
        os.makedirs(model_out, exist_ok=True)

        print(f"\n=======================================================")
        print(f" Generating 6+ Graphs for: {display_name} ({model_key})")
        print(f" Target folder: {model_out}")
        print(f"=======================================================")

        # 1. Metric Correlation Analysis
        plot_metric_correlation(df, model_key, display_name, model_out)

        # 2. Generation Latency by Dataset
        plot_generation_latency(df, model_key, display_name, model_out)

        # 3. Score Heatmap
        plot_score_heatmap(df, model_key, display_name, model_out)

        # 4. Score Distributions by Dataset
        plot_score_distributions(df, model_key, display_name, model_out)

        # 5. Multi-Metric Profile Radar
        plot_multi_metric_radar(df, model_key, display_name, model_out)

        # 6. Efficiency Frontier (Quality vs Latency)
        plot_efficiency_frontier(df, model_key, display_name, model_out)

        # 7. Dataset Quality Breakdown
        plot_dataset_quality_breakdown(df, model_key, display_name, model_out)

        # 8. Token Count by Dataset
        plot_token_count_by_dataset(df, model_key, display_name, model_out)

    # Cross-Model Comparative Suite
    comp_out = os.path.join(base_out, "cross_model_comparison")
    plot_cross_model_comparison(df, comp_out)

    print("\n[+] All model evaluation graphs successfully generated!")

if __name__ == "__main__":
    main()

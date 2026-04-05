from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
REG_DIR = PROJECT_ROOT / "data" / "outputs" / "regression"


def parse_ols_summary(path: Path) -> pd.DataFrame:
    """
    Parse coefficients and standard errors from ols_base_cluster1.txt
    and return a DataFrame with columns: name, coef, se.
    """
    lines = path.read_text().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line.startswith("prob_cluster_"):
            continue
        # Example line:
        # prob_cluster_2: coef= 1.350230, se= 0.906105, t= 1.490, p=0.156912
        name_part, rest = line.split(":", 1)
        name = name_part.strip()
        # split on commas, then parse coef and se
        parts = [p.strip() for p in rest.split(",")]
        coef_str = next(p for p in parts if p.startswith("coef="))
        se_str = next(p for p in parts if p.startswith("se="))
        coef = float(coef_str.split("=", 1)[1])
        se = float(se_str.split("=", 1)[1])
        records.append({"name": name, "coef": coef, "se": se})

    return pd.DataFrame.from_records(records)


def build_plot_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build plotting DataFrame with cluster_id, role_name, coef, ci_low, ci_high, color.
    """
    # Extract cluster_id as int from "prob_cluster_k"
    summary_df = summary_df.copy()
    summary_df["cluster_id"] = (
        summary_df["name"].str.replace("prob_cluster_", "", regex=False).astype(int)
    )

    # Mapping from cluster id to descriptive names
    cluster_name_map = {
        0: "Goalkeepers",
        1: "Quarterbacks (Right Inswingers)",
        2: "Aggressive Center Back (Left)",
        3: "Half-Space Operators (Mezzala)",
        4: "Traditional Number Nine",
        5: "Box-to-Box Midfielder (Engine)",
        6: "Pure Wingers",
        7: "Quarterbacks (Left Inswingers)",
        8: "Conservative Center Back (Right)",
        9: "Full Backs",
    }

    # 95% CI using normal approximation: coef ± 1.96 * se
    z = 1.96
    plot_df = pd.DataFrame(
        {
            "cluster_id": summary_df["cluster_id"],
            "role_name": summary_df["cluster_id"].map(cluster_name_map),
            "coef": summary_df["coef"],
            "ci_low": summary_df["coef"] - z * summary_df["se"],
            "ci_high": summary_df["coef"] + z * summary_df["se"],
        }
    )

    # Keep only clusters 2–9 (exclude GK and baseline cluster 1)
    plot_df = plot_df[plot_df["cluster_id"].between(2, 9)]

    # Determine colors based on CI sign pattern
    def classify_color(row: pd.Series) -> str:
        low, high, beta = row["ci_low"], row["ci_high"], row["coef"]
        if low > 0:
            return "#2E8B57"  # strong green (significant positive)
        if high < 0:
            return "#DC143C"  # strong red (significant negative)
        if beta > 0:
            return "#90C7A8"  # muted green (directionally positive)
        if beta < 0:
            return "#F4A582"  # muted red/orange (directionally negative)
        return "#808080"      # neutral (exact zero)

    plot_df["color"] = plot_df.apply(classify_color, axis=1)

    # Sort by coefficient descending so the most positive roles appear at the top
    plot_df = plot_df.sort_values("coef", ascending=False)
    return plot_df


def plot_coefficients(plot_df: pd.DataFrame, out_path: Path) -> None:
    """
    Create a thesis-ready coefficient (forest) plot from plot_df.
    """
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Zero line at x=0
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)

    y_positions = np.arange(len(plot_df))

    # Plot each coefficient with its CI
    for y, (_, row) in zip(y_positions, plot_df.iterrows()):
        # Horizontal error bar (CI)
        ax.errorbar(
            x=row["coef"],
            y=y,
            xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
            fmt="o",
            color=row["color"],
            ecolor=row["color"],
            elinewidth=1.5,
            capsize=3,
            markersize=6,
        )

    # Y-axis labels: role names (most positive at top)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["role_name"], fontsize=10)

    # Axis labels and title
    ax.set_xlabel("Effect on xG per 90 (coefficient)", fontsize=12)
    ax.set_ylabel("Tactical Role (cluster-based)", fontsize=12)
    ax.set_title(
        "MLR Coefficients: Tactical Role Composition vs xG per 90",
        fontsize=14,
    )

    # Faint, dotted horizontal grid lines
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="lightgrey", alpha=0.8)

    # Remove top and right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved coefficient plot to: {out_path}")


def main() -> None:
    txt_path = REG_DIR / "ols_base_cluster1.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing OLS summary file: {txt_path}")

    summary_df = parse_ols_summary(txt_path)
    plot_df = build_plot_df(summary_df)

    out_path = REG_DIR / "coef_forest_plot_base_cluster1.png"
    plot_coefficients(plot_df, out_path)


if __name__ == "__main__":
    main()


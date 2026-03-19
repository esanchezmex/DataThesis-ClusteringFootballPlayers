"""
Render cluster_stats_table.csv to a PNG image.

Input:
  - data/outputs/clusters/cluster_stats_table.csv
Output:
  - data/outputs/clusters/cluster_stats_table.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "outputs" / "clusters" / "cluster_stats_table.csv"
OUT_PATH = PROJECT_ROOT / "data" / "outputs" / "clusters" / "cluster_stats_table.png"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Formatting
    int_cols = {"cluster", "n_players", "minutes_total"}
    float_cols = [c for c in df.columns if c not in int_cols]

    df_fmt = df.copy()
    for c in int_cols:
        if c in df_fmt.columns:
            df_fmt[c] = df_fmt[c].astype(int).astype(str)
    for c in float_cols:
        df_fmt[c] = df_fmt[c].astype(float).map(lambda x: f"{x:.2f}")

    # Figure size heuristic (wide table)
    n_rows, n_cols = df_fmt.shape
    fig_w = max(10, 1.2 * n_cols)
    fig_h = max(3.5, 0.45 * (n_rows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    table = ax.table(
        cellText=df_fmt.values,
        colLabels=df_fmt.columns.tolist(),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Style cells
    header_color = "#161b22"
    even_row = "#0d1117"
    odd_row = "#0b1320"
    edge = "#30363d"
    text = "#c9d1d9"
    header_text = "#f0f6fc"

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color(header_text)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(even_row if (r % 2 == 0) else odd_row)
            cell.get_text().set_color(text)

    ax.set_title(
        "Cluster Statistics (cluster total per 90)",
        color=header_text,
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()


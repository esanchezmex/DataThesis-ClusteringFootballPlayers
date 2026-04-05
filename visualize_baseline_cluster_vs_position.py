from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_ROOT / "data" / "outputs" / "baseline_model"
AUTOENCODER_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder"


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    baseline_csv = BASELINE_DIR / "baseline_gmm_clusters.csv"
    role_cache_csv = AUTOENCODER_DIR / "player_role_cache.csv"

    if not baseline_csv.exists():
        raise FileNotFoundError(f"Missing baseline clustering CSV: {baseline_csv}")
    if not role_cache_csv.exists():
        raise FileNotFoundError(f"Missing player role cache CSV: {role_cache_csv}")

    clusters_df = pd.read_csv(baseline_csv, usecols=["player_id", "primary_cluster"])
    roles_df = pd.read_csv(role_cache_csv, usecols=["player_id", "role_name"])

    # Ensure one role per player_id (drop potential duplicates conservatively)
    roles_df = roles_df.drop_duplicates(subset=["player_id"])

    merged = clusters_df.merge(roles_df, on="player_id", how="inner")
    if merged.empty:
        raise ValueError("Merge between baseline clusters and role cache is empty.")

    # Crosstab: cluster x role (counts)
    ct = pd.crosstab(merged["primary_cluster"], merged["role_name"])

    # Sort roles alphabetically for a stable layout
    ct = ct.reindex(sorted(ct.columns), axis=1)

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.heatmap(
        ct,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt="d",
        cbar_kws={"label": "Player count"},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Actual Tactical Role (from data provider)", fontsize=12)
    ax.set_ylabel("Baseline GMM Cluster", fontsize=12)
    ax.set_title(
        "Baseline GMM: Cluster vs Tactical Role",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    out_path = BASELINE_DIR / "cluster_vs_position.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline cluster vs position heatmap to: {out_path}")


if __name__ == "__main__":
    main()


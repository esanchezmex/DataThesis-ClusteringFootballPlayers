from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
AUTOENCODER_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder"


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    clusters_csv = AUTOENCODER_DIR / "autoencoder_gmm_clusters.csv"
    role_cache_csv = AUTOENCODER_DIR / "player_role_cache.csv"

    if not clusters_csv.exists():
        raise FileNotFoundError(f"Missing autoencoder clustering CSV: {clusters_csv}")
    if not role_cache_csv.exists():
        raise FileNotFoundError(f"Missing player role cache CSV: {role_cache_csv}")

    clusters_df = pd.read_csv(clusters_csv, usecols=["player_id", "primary_cluster"])
    roles_df = pd.read_csv(role_cache_csv, usecols=["player_id", "role_name"])

    roles_df = roles_df.drop_duplicates(subset=["player_id"])

    merged = clusters_df.merge(roles_df, on="player_id", how="inner")
    if merged.empty:
        raise ValueError("Merge between autoencoder clusters and role cache is empty.")

    ct = pd.crosstab(merged["primary_cluster"], merged["role_name"])
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
    ax.set_ylabel("Autoencoder GMM Cluster", fontsize=12)
    ax.set_title(
        "Autoencoder GMM: Cluster vs Tactical Role",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    out_path = AUTOENCODER_DIR / "cluster_vs_position.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved autoencoder cluster vs position heatmap to: {out_path}")


if __name__ == "__main__":
    main()

"""
Autoencoder clusters vs actual role heatmap (baseline-style).

This replicates the baseline `cluster_vs_position.png` plot, but for the
autoencoder latent-space + GMM clustering (Option B: fit GMM on latent_* cols).

Output:
  data/outputs/autoencoder/cluster_vs_position.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ROLE_ORDER = [
    "Goalkeeper",
    "Center Back", "Left Center Back", "Right Center Back",
    "Left Back", "Right Back",
    "Left Wing Back", "Right Wing Back",
    "Defensive Midfield", "Left Defensive Midfield", "Right Defensive Midfield",
    "Left Midfield", "Right Midfield",
    "Attacking Midfield",
    "Left Winger", "Right Winger",
    "Left Forward", "Center Forward", "Right Forward",
]


def dominant_role_map(final_data_dir: Path) -> pd.DataFrame:
    """
    Returns a dataframe with columns: player_id, position (dominant role_name).
    """
    acc = None  # Series indexed by (player_id, role_name)
    for p in sorted(final_data_dir.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["player_id", "role_name"]).dropna(subset=["player_id", "role_name"])
        df["player_id"] = df["player_id"].astype("int64")
        vc = df.value_counts(subset=["player_id", "role_name"])
        acc = vc if acc is None else acc.add(vc, fill_value=0)

    acc = acc.astype("int64")
    # idxmax returns tuples (player_id, role_name) per player_id
    role_map = acc.groupby(level=0).idxmax().reset_index()
    role_map.columns = ["player_id", "_pair"]
    role_map["position"] = role_map["_pair"].apply(lambda t: t[1] if isinstance(t, tuple) and len(t) == 2 else t)
    role_map = role_map.drop(columns=["_pair"])
    return role_map


def bic_gmm_clusters(latent_df: pd.DataFrame) -> np.ndarray:
    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError("ml_ready_features_optimal.csv must contain latent_* columns for Option B.")

    Z = latent_df[latent_cols].to_numpy(dtype=np.float64)

    best_bic = float("inf")
    best_gmm = None
    for n in range(3, 11):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=3,
            random_state=42,
        )
        gmm.fit(Z)
        bic = gmm.bic(Z)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    return best_gmm.predict(Z)


def main() -> None:
    clusters_path = PROJECT_ROOT / "data" / "outputs" / "autoencoder" / "ml_ready_features_optimal.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Missing: {clusters_path}")

    latent_df = pd.read_csv(clusters_path)
    if "player_id" not in latent_df.columns:
        raise ValueError("ml_ready_features_optimal.csv must contain player_id")

    with open(CREDS_FILE) as f:
        cfg = json.load(f)
    final_data_dir = Path(cfg["final_data"])
    if not final_data_dir.exists():
        raise FileNotFoundError(f"final_data directory not found: {final_data_dir}")

    primary_cluster = bic_gmm_clusters(latent_df)
    clusters = pd.DataFrame({"player_id": latent_df["player_id"].astype("int64"), "primary_cluster": primary_cluster})

    roles_df = dominant_role_map(final_data_dir)
    merged = clusters.merge(roles_df, on="player_id", how="left")
    merged["position"] = merged["position"].fillna("Unknown")

    # Crosstab
    present_roles = [r for r in ROLE_ORDER if r in merged["position"].unique()]
    extra_roles = [r for r in merged["position"].unique() if r not in ROLE_ORDER]
    role_order = present_roles + sorted(extra_roles)

    ct = pd.crosstab(merged["primary_cluster"], merged["position"])
    ct = ct.reindex(columns=role_order, fill_value=0).sort_index()

    # Plot (baseline-style): raw counts, cap color scale at 30
    fig_w = max(14, len(role_order) * 0.85)
    fig_h = max(5, len(ct.index) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=30,
        linewidths=0.5,
        linecolor="#cccccc",
        cbar_kws={"label": "Player count", "shrink": 0.7},
        ax=ax,
    )

    ax.set_title(
        "Autoencoder Latent GMM – Cluster vs Actual Tactical Role\n(raw player counts, colour scale capped at 30)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Tactical Role (from data provider)", fontsize=10)
    ax.set_ylabel("Primary Cluster", fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "cluster_vs_position.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


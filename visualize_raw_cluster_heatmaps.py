"""
Ground-truth cluster average heatmaps (raw spatial tensors)
----------------------------------------------------------

This script aggregates the *raw* player spatial tensors by cluster assignment and
plots mean Layer 0 (Presence) per cluster on an mplsoccer pitch.

Inputs:
  - Cluster assignments: data/outputs/autoencoder_tuning/ml_ready_features_optimal.csv
    (If missing, falls back to data/outputs/autoencoder/ml_ready_features_optimal.csv)
    If the CSV does not include primary_cluster, this script will fit a GMM on the
    latent_* columns (BIC sweep 3–10) and create primary_cluster.
  - Player profiles: creds/gdrive_folder.json -> final_data sibling -> player_spatial_profiles/processed_player_profiles.pkl

Output:
  - data/outputs/autoencoder_tuning/raw_average_heatmaps_per_cluster.png
"""

import json
import math
from pathlib import Path
from typing import Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_inputs() -> Tuple[Path, Path]:
    """
    Returns:
      clusters_csv, profiles_pkl
    """
    clusters_csv_preferred = PROJECT_ROOT / "data" / "outputs" / "autoencoder_tuning" / "ml_ready_features_optimal.csv"
    clusters_csv_fallback = PROJECT_ROOT / "data" / "outputs" / "autoencoder" / "ml_ready_features_optimal.csv"

    clusters_csv = clusters_csv_preferred if clusters_csv_preferred.exists() else clusters_csv_fallback
    if not clusters_csv.exists():
        raise FileNotFoundError(
            "Could not find ml_ready_features_optimal.csv at either:\n"
            f"  - {clusters_csv_preferred}\n"
            f"  - {clusters_csv_fallback}"
        )

    with open(CREDS_FILE) as f:
        cfg = json.load(f)
    final_data_dir = Path(cfg["final_data"])
    profiles_pkl = final_data_dir.parent / "player_spatial_profiles" / "processed_player_profiles.pkl"
    if not profiles_pkl.exists():
        raise FileNotFoundError(f"Missing player profiles pkl: {profiles_pkl}")

    return clusters_csv, profiles_pkl


def ensure_primary_cluster(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure clusters_df has a primary_cluster column.
    If missing, fit a BIC-selected GMM on latent_* columns and create it.
    """
    if "primary_cluster" in clusters_df.columns:
        return clusters_df

    latent_cols = [c for c in clusters_df.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError(
            "Cluster CSV is missing 'primary_cluster' and has no latent_* columns to fit a GMM."
        )

    Z = clusters_df[latent_cols].to_numpy(dtype=np.float64)

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

    clusters_df = clusters_df.copy()
    clusters_df["primary_cluster"] = best_gmm.predict(Z)
    return clusters_df


def main() -> None:
    clusters_csv, profiles_pkl = resolve_inputs()
    print(f"Loading clusters CSV: {clusters_csv}")
    clusters_df = pd.read_csv(clusters_csv)
    clusters_df = ensure_primary_cluster(clusters_df)

    needed_cols = {"player_id", "primary_cluster"}
    if not needed_cols.issubset(set(clusters_df.columns)):
        raise ValueError(f"CSV must contain {sorted(needed_cols)}. Found: {clusters_df.columns.tolist()}")

    print(f"Loading player profiles PKL: {profiles_pkl}")
    profiles_df = pd.read_pickle(profiles_pkl)
    if "player_id" not in profiles_df.columns or "spatial_tensor" not in profiles_df.columns:
        raise ValueError("processed_player_profiles.pkl must contain player_id and spatial_tensor columns.")

    merged = profiles_df.merge(
        clusters_df[["player_id", "primary_cluster"]],
        on="player_id",
        how="inner",
    )
    print(f"Merged players: {len(merged)}")

    clusters = sorted(merged["primary_cluster"].unique())
    n_clusters = len(clusters)
    print(f"Clusters: {n_clusters} -> {clusters}")

    # Dynamic grid (e.g., 10 clusters -> 2x5)
    n_cols = 5 if n_clusters >= 8 else 3
    n_rows = math.ceil(n_clusters / n_cols)

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#1e1e1e",
        line_color="#ffffff",
        linewidth=1.5,
        goal_type="box",
    )

    fig, axes = pitch.draw(nrows=n_rows, ncols=n_cols, figsize=(24, 12))
    fig.patch.set_facecolor("#1e1e1e")
    axes_flat = axes.flatten() if (n_rows * n_cols) > 1 else [axes]

    for i, cid in enumerate(clusters):
        ax = axes_flat[i]
        sub = merged[merged["primary_cluster"] == cid]
        n_players = len(sub)

        tensors = np.stack(sub["spatial_tensor"].values)  # (n, 5, 50, 50)
        mean_tensor = tensors.mean(axis=0)                # (5, 50, 50)

        heatmap = mean_tensor[0]  # Layer 0 (Presence), shape (50, 50)
        heatmap = heatmap.T       # transpose for imshow alignment (rows=y, cols=x)
        heatmap = gaussian_filter(heatmap, sigma=1.0)

        ax.imshow(
            heatmap,
            extent=[0, 105, 0, 68],
            origin="lower",
            cmap="magma",
            alpha=0.85,
            aspect="auto",
            zorder=2,
        )

        ax.set_title(f"Cluster {cid} (n={n_players} players)", fontsize=16, color="white", fontweight="bold", pad=8)

    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=3.0)
    out_path = OUTPUT_DIR / "raw_average_heatmaps_per_cluster.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


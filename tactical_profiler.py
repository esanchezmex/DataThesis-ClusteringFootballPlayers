"""
Tactical Profiler (cluster-level layer visualization)
----------------------------------------------------

Creates a 1x5 row of pitch heatmaps for a chosen cluster, showing the mean
spatial tensor per layer across all players assigned to that cluster.

Inputs:
  - Cluster assignments: data/outputs/autoencoder/ml_ready_features_optimal.csv
    (If primary_cluster is missing, we fit a BIC-selected GMM on latent_* columns.)
  - Player profiles: creds/gdrive_folder.json -> final_data sibling ->
    player_spatial_profiles/processed_player_profiles.pkl

Output:
  - data/outputs/clusters/cluster_<k>_tactical_profile.png
"""

from __future__ import annotations

import argparse
import json
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
AUTOENCODER_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder"
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9

def resolve_inputs() -> Tuple[Path, Path]:
    clusters_csv = AUTOENCODER_DIR / "ml_ready_features_optimal.csv"
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Missing clusters/features CSV: {clusters_csv}")

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
        raise ValueError("CSV missing primary_cluster and no latent_* columns available to fit a GMM.")

    Z = clusters_df[latent_cols].to_numpy(dtype=np.float64)

    best_bic = float("inf")
    best_gmm: GaussianMixture | None = None
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

    out = clusters_df.copy()
    out["primary_cluster"] = best_gmm.predict(Z)
    return out


def _to_prob_maps(tensors: np.ndarray) -> np.ndarray:
    """
    Convert raw count tensors to per-player, per-layer probability maps.
    tensors: (n, 5, 50, 50)
    """
    t = tensors.astype(np.float32, copy=True)
    denom = t.sum(axis=(2, 3), keepdims=True)
    return np.divide(t, denom, out=np.zeros_like(t), where=denom > 0)


def _robust_vmax(img: np.ndarray, percentile: float = 99.5) -> float:
    vmax = float(np.percentile(img, percentile))
    if vmax <= 0:
        vmax = float(img.max()) or 1.0
    return vmax


def plot_cluster_layers(
    cluster_mean: np.ndarray,
    global_mean: np.ndarray,
    cluster_id: int,
    n_players: int,
    smoothing_sigma: float,
    mode: str,
) -> Path:
    """
    cluster_mean/global_mean: shape (5, 50, 50) in (layer, x_bin, y_bin)
    mode:
      - "mean": cluster mean probability maps
      - "diff": (cluster - global) difference maps (diverging)
      - "logratio": log((cluster+eps)/(global+eps)) maps (diverging)
    """
    if cluster_mean.shape != (5, 50, 50) or global_mean.shape != (5, 50, 50):
        raise ValueError(f"Expected (5,50,50) means, got cluster={cluster_mean.shape} global={global_mean.shape}")
    plt.rcParams["font.family"] = "Times New Roman"

    # Layer naming: these reflect what build_player_spatial_profiles.py currently encodes.
    layer_titles: List[str] = [
        "Layer 0: Offensive presence",
        "Layer 1: Pass locations",
        "Layer 2: Carries / dribbles (proxy for runs)",
        "Layer 3: Goal threat (shots + key passes)",
        "Layer 4: Receptions (ball receipt)",
    ]

    if mode == "mean":
        title_suffix = "Cluster mean (per-player probability maps)"
        cmap = "magma"
        diverging = False
    elif mode == "diff":
        title_suffix = "Cluster − Global (difference)"
        cmap = "coolwarm"
        diverging = True
    elif mode == "logratio":
        title_suffix = "log(Cluster / Global)"
        cmap = "coolwarm"
        diverging = True
    else:
        raise ValueError("mode must be one of: mean, diff, logratio")

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="white",
        line_color="black",
        linewidth=1.5,
        goal_type="box",
    )

    # Avoid suptitle overlapping subplot titles by reserving top margin manually.
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2), constrained_layout=False)
    fig.patch.set_facecolor("white")

    # Prepare all layers and share a per-layer robust vmax
    last_im = None
    for i in range(5):
        ax = axes[i]
        pitch.draw(ax=ax)
        ax.set_facecolor("white")

        if mode == "mean":
            layer = cluster_mean[i]
        elif mode == "diff":
            layer = cluster_mean[i] - global_mean[i]
        else:  # logratio
            layer = np.log((cluster_mean[i] + EPS) / (global_mean[i] + EPS))

        layer_img = gaussian_filter(layer.T, sigma=smoothing_sigma)  # transpose -> (y,x)

        if diverging:
            # Asymmetric scaling to emphasize positive deviations (reds) over negative (blues).
            # Compute robust caps separately on positive/negative tails.
            pos = layer_img[layer_img > 0]
            neg = -layer_img[layer_img < 0]  # magnitude of negatives

            vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
            vneg = float(np.percentile(neg, 99.5)) if neg.size else vmax

            # Compress negative range so blues don't dominate the map.
            NEG_SCALE = 0.45  # lower -> more emphasis on positives
            vmin = -max(vneg * NEG_SCALE, 1e-6)
        else:
            vmax = _robust_vmax(layer_img, percentile=99.5)
            vmin = 0.0

        last_im = ax.imshow(
            layer_img,
            extent=[0, 105, 0, 68],
            origin="lower",
            cmap=cmap,
            alpha=0.88,
            aspect="auto",
            zorder=2,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(layer_titles[i], fontsize=10, color="black", fontweight="bold", pad=8)

    # Add a global attacking-direction arrow (left → right) on the first subplot,
    # using axes coordinates so it is always visible just above the top edge.
    arrow_ax = axes[0]
    arrow_ax.annotate(
        "",
        xy=(0.9, 1.18),      # end point (right)
        xytext=(0.1, 1.18),  # start point (left)
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=2),
        clip_on=False,
    )
    arrow_ax.text(
        0.5,
        1.22,
        "Attacking direction",
        transform=arrow_ax.transAxes,
        color="black",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )

    fig.suptitle(
        f"Tactical Profiler — Cluster {cluster_id} (n={n_players} players)\n{title_suffix}",
        color="black",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Shared legend (colorbar) for the 5 facets (horizontal at bottom)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.12)
        if mode == "mean":
            cbar.set_label("probability mass (per-player, per-layer)", color="black")
        elif mode == "diff":
            cbar.set_label("Δ probability mass (cluster − global)", color="black")
        else:  # logratio
            cbar.set_label("log(cluster / global)", color="black")
        cbar.ax.xaxis.set_tick_params(color="black", labelsize=10)
        cbar.ax.xaxis.label.set_size(12)
        for tick in cbar.ax.get_xticklabels():
            tick.set_color("black")

    out_path = OUT_DIR / f"cluster_{cluster_id}_tactical_profile_{mode}.png"
    # Leave space at top for the suptitle and at bottom for the horizontal colorbar.
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=int, default=0, help="Cluster id to profile (default: 0).")
    parser.add_argument("--mode", type=str, default="mean", choices=["mean", "diff", "logratio"], help="Visualisation mode.")
    parser.add_argument("--sigma", type=float, default=1.6, help="Gaussian smoothing sigma for display.")
    args = parser.parse_args()

    clusters_csv, profiles_pkl = resolve_inputs()

    clusters_df = pd.read_csv(clusters_csv)
    clusters_df = ensure_primary_cluster(clusters_df)

    if "player_id" not in clusters_df.columns:
        raise ValueError("ml_ready_features_optimal.csv must contain player_id")

    profiles_df = pd.read_pickle(profiles_pkl)
    if not {"player_id", "spatial_tensor"}.issubset(set(profiles_df.columns)):
        raise ValueError("processed_player_profiles.pkl must contain player_id and spatial_tensor")

    merged = profiles_df.merge(
        clusters_df[["player_id", "primary_cluster"]],
        on="player_id",
        how="inner",
    )

    sub = merged[merged["primary_cluster"] == args.cluster]
    if sub.empty:
        available = sorted(merged["primary_cluster"].unique().tolist())
        raise ValueError(f"No players found for cluster={args.cluster}. Available clusters: {available}")

    # Use per-player probability maps to reduce "busy player" dominance and highlight shapes.
    cluster_tensors = np.stack(sub["spatial_tensor"].values)  # (n, 5, 50, 50)
    global_tensors = np.stack(merged["spatial_tensor"].values)  # (N, 5, 50, 50)

    cluster_prob = _to_prob_maps(cluster_tensors)
    global_prob = _to_prob_maps(global_tensors)

    cluster_mean = cluster_prob.mean(axis=0)
    global_mean = global_prob.mean(axis=0)

    out = plot_cluster_layers(
        cluster_mean=cluster_mean,
        global_mean=global_mean,
        cluster_id=args.cluster,
        n_players=len(sub),
        smoothing_sigma=float(args.sigma),
        mode=str(args.mode),
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


import json
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"
OUT_PATH = PROJECT_ROOT / "data" / "outputs" / "autoencoder" / "raw_layer0_sanitycheck.png"


def resolve_profiles_pkl() -> Path:
    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)
    final_data_dir = Path(cfg["final_data"])
    profiles_pkl = final_data_dir.parent / "player_spatial_profiles" / "processed_player_profiles.pkl"
    if not profiles_pkl.exists():
        raise FileNotFoundError(f"Missing player profiles pkl: {profiles_pkl}")
    return profiles_pkl


def layer0_stats(layer0: np.ndarray) -> Tuple[float, float, Tuple[int, int, int, int]]:
    """
    layer0: (50, 50) in histogram bin space (x_bins, y_bins)
    Returns: (max, p98, corner_counts)
    """
    arr = np.asarray(layer0, dtype=np.float32)
    max_v = float(np.max(arr)) if arr.size else 0.0
    p98 = float(np.percentile(arr, 98)) if arr.size else 0.0
    corners = (
        int(arr[0, 0]),
        int(arr[0, -1]),
        int(arr[-1, 0]),
        int(arr[-1, -1]),
    )
    return max_v, p98, corners


def main() -> None:
    """
    Sanity check: sample a handful of players and visualize RAW layer-0 (presence)
    tensors from processed_player_profiles.pkl.

    This is meant to catch obvious issues like "double-lobed" players (halves not
    aligned) or border artifacts.
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles_pkl = resolve_profiles_pkl()

    df = pd.read_pickle(profiles_pkl)
    if "player_id" not in df.columns or "spatial_tensor" not in df.columns:
        raise ValueError("processed_player_profiles.pkl must contain player_id and spatial_tensor columns.")

    # Reproducible sample (6 players)
    n_show = min(6, len(df))
    sample = df.sample(n=n_show, random_state=42).reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 4.8))
    fig.patch.set_facecolor("#0d1117")

    for i in range(6):
        ax = axes.flat[i]
        ax.set_facecolor("#0d1117")
        ax.set_xticks([])
        ax.set_yticks([])

        if i >= n_show:
            ax.axis("off")
            continue

        pid = int(sample.loc[i, "player_id"])
        tensor = sample.loc[i, "spatial_tensor"]
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.shape[0] < 1:
            ax.set_title(f"player_id={pid}\n(empty tensor)", color="white", fontsize=9)
            continue

        layer0 = tensor[0]  # (50, 50) in (x_bins, y_bins)
        max_v, p98, corners = layer0_stats(layer0)

        # For imshow on a pitch-like canvas, transpose so rows=y and cols=x
        img = layer0.T
        vmax = p98 if p98 > 0 else (max_v if max_v > 0 else 1.0)

        ax.imshow(img, cmap="magma", origin="lower", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(
            f"player_id={pid}\nmax={max_v:.1f}  p98={p98:.1f}  corners={list(corners)}",
            color="white",
            fontsize=9,
            pad=6,
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle(
        "RAW Layer 0 (Presence) from processed_player_profiles.pkl",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(OUT_PATH, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print(f"Loaded: {profiles_pkl}")


if __name__ == "__main__":
    main()


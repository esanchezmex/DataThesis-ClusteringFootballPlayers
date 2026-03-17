import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"

# Centered SkillCorner grass bounds (for plotting sanity / filtering extreme noise only)
X_MIN, X_MAX = -52.5, 52.5
Y_MIN, Y_MAX = -34.0, 34.0


def main() -> None:
    """
    Reset sanity check: plot RAW goalkeeper locations from a single match parquet,
    with NO attack normalization and NO half flipping.

    Notes:
      - We assume raw coordinates are centered ([-52.5, 52.5] x [-34, 34]).
      - For plotting on a 105x68 pitch, we only SHIFT (x+52.5, y+34.0).
    """
    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)

    final_data_dir = Path(cfg["final_data"])
    parquet_files = sorted(final_data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {final_data_dir}")

    match_path = parquet_files[0]
    df = pd.read_parquet(match_path)

    role_col = "role_name" if "role_name" in df.columns else ("position" if "position" in df.columns else None)
    if role_col is None:
        raise ValueError("Could not find a role/position column (expected 'role_name' or 'position').")

    required = {"player_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Filter GK rows
    gk = df[df[role_col].astype(str).str.lower().eq("goalkeeper")].copy()
    if gk.empty:
        raise ValueError(f"No goalkeepers found in {match_path.name}.")

    gk["x"] = pd.to_numeric(gk["x"], errors="coerce")
    gk["y"] = pd.to_numeric(gk["y"], errors="coerce")
    gk = gk.dropna(subset=["player_id", "x", "y"])

    # Drop extreme out-of-bounds points (optical noise), but do not normalize/flip
    in_bounds = (gk["x"].between(X_MIN, X_MAX)) & (gk["y"].between(Y_MIN, Y_MAX))
    gk = gk[in_bounds].copy()

    # Pick GK with most samples
    chosen_pid = int(gk.groupby("player_id").size().sort_values(ascending=False).index[0])
    sub = gk[gk["player_id"] == chosen_pid].copy()

    # Shift to pitch coordinates for plotting only
    x_plot = sub["x"].to_numpy() + 52.5
    y_plot = sub["y"].to_numpy() + 34.0

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#1e1e1e",
        line_color="#ffffff",
        linewidth=1.5,
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=(10, 6))
    fig.patch.set_facecolor("#1e1e1e")

    x_edges = np.linspace(0.0, 105.0, 80)
    y_edges = np.linspace(0.0, 68.0, 55)
    h, _, _ = np.histogram2d(x_plot, y_plot, bins=[x_edges, y_edges])
    h = h.T

    im = ax.imshow(
        np.log1p(h),
        extent=[0.0, 105.0, 0.0, 68.0],
        origin="lower",
        cmap="magma",
        alpha=0.9,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log(1 + count)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    match_id = (
        int(df["match_id"].dropna().iloc[0])
        if "match_id" in df.columns and df["match_id"].notna().any()
        else match_path.stem
    )
    ax.set_title(
        f"RAW GK location heatmap (no normalization)\nmatch={match_id} | player_id={chosen_pid}",
        fontsize=11,
        color="white",
        pad=12,
        fontweight="bold",
    )

    out_path = PROJECT_ROOT / "raw_gk_heatmap_sanity_check.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Used file: {match_path}")
    print(f"Chosen GK player_id: {chosen_pid}")
    print(f"Rows plotted: {len(sub)}")


if __name__ == "__main__":
    main()


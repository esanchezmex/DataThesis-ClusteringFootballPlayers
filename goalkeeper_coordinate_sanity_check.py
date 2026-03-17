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


def main() -> None:
    # 1) Load config + locate final_data (our raw merged tracking/event table)
    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)
    final_data_dir = Path(cfg["final_data"])
    if not final_data_dir.exists():
        raise FileNotFoundError(f"final_data directory not found: {final_data_dir}")

    # 2) Pick a single match parquet (first file) and load only needed columns
    parquet_files = sorted(final_data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {final_data_dir}")

    match_path = parquet_files[0]
    cols = ["match_id", "player_id", "role_name", "position", "period", "team_id", "x", "y"]
    df = pd.read_parquet(match_path, columns=[c for c in cols if c in pd.read_parquet(match_path, engine="pyarrow").columns])

    # Fallback if role_name not present
    role_col = "role_name" if "role_name" in df.columns else ("position" if "position" in df.columns else None)
    if role_col is None:
        raise ValueError("Could not find a role/position column (expected 'role_name' or 'position').")

    needed = {"player_id", "period", "x", "y", "match_id", "team_id"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing required columns in parquet: {sorted(needed - set(df.columns))}")

    # --- Apply the SAME x_norm/y_norm logic as the profile builder (quantile-based flip) ---
    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df["team_defending_x"] = df.groupby(["match_id", "period", "team_id"])["x"].transform(lambda val: val.quantile(0.10))
    flip_mask = df["team_defending_x"] > 0

    df["x_norm"] = df["x"].copy()
    df["y_norm"] = df["y"].copy()
    df.loc[flip_mask, "x_norm"] = df.loc[flip_mask, "x"] * -1
    df.loc[flip_mask, "y_norm"] = df.loc[flip_mask, "y"] * -1

    valid_x = (df["x_norm"] >= -52.5) & (df["x_norm"] <= 52.5)
    valid_y = (df["y_norm"] >= -34.0) & (df["y_norm"] <= 34.0)
    df = df[valid_x & valid_y].copy()
    df = df.drop(columns=["team_defending_x"])

    # 3) Filter for goalkeepers
    gk_df = df[df[role_col].astype(str).str.lower().eq("left center back")].dropna(subset=["x_norm", "y_norm", "period", "player_id"])
    if gk_df.empty:
        raise ValueError(f"No goalkeepers found in {match_path.name} using column {role_col}.")

    # Pick one goalkeeper who appears in both halves with lots of samples
    candidates = (
        gk_df.groupby("player_id")["period"]
        .agg(lambda s: set(s.unique()))
        .reset_index(name="periods")
    )
    candidates = candidates[candidates["periods"].apply(lambda s: 1 in s and 2 in s)]
    if candidates.empty:
        raise ValueError("No goalkeeper with both period 1 and 2 present in this match file.")

    # Choose the GK with the most rows (proxy for full match participation)
    row_counts = gk_df.groupby("player_id").size().rename("n_rows").reset_index()
    candidates = candidates.merge(row_counts, on="player_id", how="left").sort_values("n_rows", ascending=False)
    chosen_player_id = int(candidates.iloc[0]["player_id"])

    plot_df = gk_df[gk_df["player_id"] == chosen_player_id].copy()
    match_id = int(plot_df["match_id"].dropna().iloc[0]) if "match_id" in plot_df.columns and plot_df["match_id"].notna().any() else None

    # 4) Plot on pitch
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

    # 5) Scatter by half
    first = plot_df[plot_df["period"] == 1]
    second = plot_df[plot_df["period"] == 2]

    # Convert centered coords to pitch coords for plotting
    # Apply 180° rotation to 1st half only (centered coords)
    fx = (first["x_norm"] * -1) + 52.5
    fy = (first["y_norm"] * -1) + 34.0
    sx = second["x_norm"] + 52.5
    sy = second["y_norm"] + 34.0

    ax.scatter(fx, fy, s=6, c="#1f77b4", alpha=0.45, label="1st Half (x_norm/y_norm)")
    ax.scatter(sx, sy, s=6, c="#d62728", alpha=0.45, label="2nd Half (x_norm/y_norm)")

    # 6) Title
    title_match = f"match_id={match_id}" if match_id is not None else match_path.stem
    ax.set_title(
        f"Goalkeeper Attack-Normalized Coordinates: 1st Half (Blue) vs 2nd Half (Red)\n{title_match} | player_id={chosen_player_id}",
        fontsize=12,
        color="white",
        pad=12,
        fontweight="bold",
    )
    leg = ax.legend(framealpha=0.4, facecolor="#1e1e1e", edgecolor="white", fontsize=10)
    for text in leg.get_texts():
        text.set_color("white")

    # 7) Save
    out_path = PROJECT_ROOT / "goalkeeper_sanity_check.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Used file: {match_path}")
    print(f"Selected player_id: {chosen_player_id} (rows={len(plot_df)})")


if __name__ == "__main__":
    main()


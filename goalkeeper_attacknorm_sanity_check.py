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

# Centered SkillCorner grass bounds
X_MIN, X_MAX = -52.5, 52.5
Y_MIN, Y_MAX = -34.0, 34.0


def main() -> None:
    """
    Sanity check:
      - Coordinates are in centered stadium frame (roughly [-52.5,52.5] × [-34,34]).
      - Normalise attacking direction so the chosen GK's team always defends -X (attacks +X).
      - Plot 1H vs 2H on a standard 105×68 pitch (we shift centered coords to 0..105/0..68 for drawing).
    """
    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)
    final_data_dir = Path(cfg["final_data"])
    parquet_files = sorted(final_data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {final_data_dir}")

    match_path = parquet_files[0]

    # Load minimal columns
    wanted = ["match_id", "player_id", "role_name", "position", "period", "x", "y", "team_id", "team", "event_team"]
    df = pd.read_parquet(match_path)
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()

    # Role/position column
    role_col = "role_name" if "role_name" in df.columns else ("position" if "position" in df.columns else None)
    if role_col is None:
        raise ValueError("Could not find a role/position column (expected 'role_name' or 'position').")

    # Team key (stable if present)
    if "team_id" in df.columns:
        team_key = "team_id"
    elif "team" in df.columns:
        team_key = "team"
    elif "event_team" in df.columns:
        team_key = "event_team"
    else:
        raise ValueError("Could not find a team identifier (team_id/team/event_team).")

    # Filter GK rows for candidate selection
    gk = df[df[role_col].astype(str).str.lower().eq("goalkeeper")].dropna(subset=["player_id", "period", "x", "y", team_key])
    if gk.empty:
        raise ValueError(f"No goalkeepers found in {match_path.name}.")

    # Pick GK with both halves, most samples
    have_both = gk.groupby("player_id")["period"].agg(lambda s: set(s.unique()))
    have_both = have_both[have_both.apply(lambda s: 1 in s and 2 in s)]
    if have_both.empty:
        raise ValueError("No goalkeeper with both halves present in this match.")
    counts = gk.groupby("player_id").size()
    chosen_pid = int(counts.loc[have_both.index].sort_values(ascending=False).index[0])

    gk_pid = gk[gk["player_id"] == chosen_pid].copy()
    match_id = int(gk_pid["match_id"].dropna().iloc[0]) if "match_id" in gk_pid.columns and gk_pid["match_id"].notna().any() else match_path.stem

    # Compute defending side (median x) per (period, team_key), prefer GK median else team median
    # This mirrors the logic used in build_player_spatial_profiles.py.
    grp = df.groupby(["period", team_key], sort=False)
    team_med_x = grp["x"].median()
    gk_med_x = gk.groupby(["period", team_key], sort=False)["x"].median()

    defend_med_x = team_med_x.copy()
    defend_med_x.loc[gk_med_x.index] = gk_med_x

    # Flip if defending median x is positive (defending +X)
    flip_groups = set(defend_med_x[defend_med_x > 0].index.tolist())

    def _flip_row(row) -> bool:
        return (row["period"], row[team_key]) in flip_groups

    flip_flag = gk_pid.apply(_flip_row, axis=1)
    gk_pid["_x_norm"] = np.where(flip_flag, -pd.to_numeric(gk_pid["x"], errors="coerce"), pd.to_numeric(gk_pid["x"], errors="coerce"))
    gk_pid["_y_norm"] = np.where(flip_flag, -pd.to_numeric(gk_pid["y"], errors="coerce"), pd.to_numeric(gk_pid["y"], errors="coerce"))

    # Drop out-of-bounds (no clipping)
    in_bounds = (
        (gk_pid["_x_norm"] >= X_MIN) & (gk_pid["_x_norm"] <= X_MAX) &
        (gk_pid["_y_norm"] >= Y_MIN) & (gk_pid["_y_norm"] <= Y_MAX)
    )
    gk_pid = gk_pid[in_bounds].copy()

    # Shift for plotting on 0..105 / 0..68 pitch
    gk_pid["_x_plot"] = gk_pid["_x_norm"] + 52.5
    gk_pid["_y_plot"] = gk_pid["_y_norm"] + 34.0

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

    first = gk_pid[gk_pid["period"] == 1]
    second = gk_pid[gk_pid["period"] == 2]

    ax.scatter(first["_x_plot"], first["_y_plot"], s=7, c="#1f77b4", alpha=0.55, label="1st Half (normed)")
    ax.scatter(second["_x_plot"], second["_y_plot"], s=7, c="#d62728", alpha=0.55, label="2nd Half (normed)")

    ax.set_title(
        "Goalkeeper Attack-Normalized Coordinates: 1st Half (Blue) vs 2nd Half (Red)\n"
        f"{match_id} | player_id={chosen_pid} | centered coords → shifted to pitch for plotting",
        fontsize=11,
        color="white",
        pad=12,
        fontweight="bold",
    )
    leg = ax.legend(framealpha=0.4, facecolor="#1e1e1e", edgecolor="white", fontsize=9)
    for text in leg.get_texts():
        text.set_color("white")

    out_path = PROJECT_ROOT / "goalkeeper_attacknorm_sanity_check.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Used file: {match_path}")
    print(f"Selected player_id: {chosen_pid}")


if __name__ == "__main__":
    main()


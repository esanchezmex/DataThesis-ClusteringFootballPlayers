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

# Trust GK median only if clearly away from midfield
GK_TRUST_THRESH = 30.0


def main() -> None:
    """
    Diagnostic: for one match, infer defending-side per (team_id, period) using:
      - GK median-x if |median-x| >= GK_TRUST_THRESH
      - else fallback to team 10th-percentile x

    Then plot RAW GK locations (no normalization) and mark GK median points for each
    (team_id, period) on a pitch so we can visually confirm which goal they defend.
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

    team_col = "team" if "team" in df.columns else ("team_id" if "team_id" in df.columns else None)
    if team_col is None:
        raise ValueError("Could not find a stable team column (expected 'team' or 'team_id').")

    required = {"match_id", "period", team_col, "player_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["match_id", "period", team_col, "player_id", "x", "y"])

    # Filter to grass bounds to avoid optical junk dominating
    df = df[df["x"].between(X_MIN, X_MAX) & df["y"].between(Y_MIN, Y_MAX)].copy()

    gk = df[df[role_col].astype(str).str.lower().eq("goalkeeper")].copy()
    if gk.empty:
        raise ValueError(f"No goalkeepers found in {match_path.name}.")

    group_keys = ["match_id", "period", team_col]
    team_q10_x = df.groupby(group_keys, sort=False)["x"].quantile(0.10).rename("team_q10_x")
    gk_med_x = gk.groupby(group_keys, sort=False)["x"].median().rename("gk_med_x")
    gk_med_y = gk.groupby(group_keys, sort=False)["y"].median().rename("gk_med_y")

    # Combine into one table (index = group_keys)
    diag = pd.concat([team_q10_x, gk_med_x, gk_med_y], axis=1).reset_index()
    diag["gk_trusted"] = diag["gk_med_x"].abs() >= GK_TRUST_THRESH
    diag["defend_x_used"] = np.where(diag["gk_trusted"], diag["gk_med_x"], diag["team_q10_x"])
    diag["defending_plus_x"] = diag["defend_x_used"] > 0

    match_id = int(df["match_id"].iloc[0])
    out_csv = PROJECT_ROOT / "defending_side_diagnostic.csv"
    diag.to_csv(out_csv, index=False)

    # Plot raw GK locations + median markers by (team_id, period)
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#1e1e1e",
        line_color="#ffffff",
        linewidth=1.5,
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=(11, 6.5))
    fig.patch.set_facecolor("#1e1e1e")

    teams = sorted(gk[team_col].unique().tolist())
    if len(teams) > 2:
        teams = teams[:2]

    team_colors = {teams[0]: "#1f77b4"}
    if len(teams) > 1:
        team_colors[teams[1]] = "#d62728"

    # Raw points (small)
    for team_id in teams:
        for period, marker in [(1, "o"), (2, "^")]:
            sub = gk[(gk[team_col] == team_id) & (gk["period"] == period)]
            if sub.empty:
                continue
            x_plot = sub["x"].to_numpy() + 52.5
            y_plot = sub["y"].to_numpy() + 34.0
            ax.scatter(
                x_plot,
                y_plot,
                s=6,
                alpha=0.18,
                c=team_colors.get(team_id, "white"),
                marker=marker,
                linewidths=0,
            )

    # Median markers (big X) + annotation
    for _, row in diag.iterrows():
        team_id = int(row[team_col])
        if team_id not in team_colors:
            continue
        period = int(row["period"])
        mx = float(row["gk_med_x"])
        my = float(row["gk_med_y"])
        ax.scatter(
            mx + 52.5,
            my + 34.0,
            s=180,
            marker="X",
            c=team_colors[team_id],
            edgecolors="white",
            linewidths=1.0,
            alpha=0.95,
            zorder=5,
        )
        used = float(row["defend_x_used"])
        src = "GKmed" if bool(row["gk_trusted"]) else "q10"
        ax.text(
            mx + 52.5,
            my + 34.0 + (2.0 if period == 1 else -2.0),
            f"{team_col}={team_id} p{period}\n{src}={used:+.1f}",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=6,
        )

    ax.set_title(
        f"Defending-side diagnostic (RAW GK points) | match={match_id}\n"
        f"Median markers are GK med (X). Labels show defend_x_used (GKmed if trusted else team q10).",
        fontsize=11,
        color="white",
        pad=12,
        fontweight="bold",
    )

    # Legend (team colors + half markers)
    handles = []
    labels = []
    for team_id in teams:
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=team_colors[team_id], markersize=8))
        labels.append(f"{team_col}={team_id} (color)")
    handles.append(plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="white", markersize=7, linestyle="None"))
    labels.append("period=1 (circle)")
    handles.append(plt.Line2D([0], [0], marker="^", color="white", markerfacecolor="white", markersize=7, linestyle="None"))
    labels.append("period=2 (triangle)")
    ax.legend(handles, labels, framealpha=0.35, facecolor="#1e1e1e", edgecolor="white", fontsize=8, loc="upper left")

    out_png = PROJECT_ROOT / "defending_side_diagnostic_pitch.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved table: {out_csv}")
    print(f"Used file: {match_path}")


if __name__ == "__main__":
    main()


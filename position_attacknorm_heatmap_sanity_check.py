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


def attack_normalize_like_gk_sanity(df: pd.DataFrame) -> pd.DataFrame:
    team_col = "team" if "team" in df.columns else ("team_id" if "team_id" in df.columns else None)
    if team_col is None:
        raise ValueError("Could not find a stable team column (expected 'team' or 'team_id').")

    required_cols = {"match_id", "period", team_col, "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    group_keys = ["match_id", "period", team_col]
    team_q10_x = out.groupby(group_keys, sort=False)["x"].quantile(0.10)
    defend_x = team_q10_x.copy()

    role_col = "role_name" if "role_name" in out.columns else ("position" if "position" in out.columns else None)
    if role_col is not None:
        gk_rows = out[out[role_col].astype(str).str.lower().eq("goalkeeper")]
        if not gk_rows.empty:
            gk_med_x = gk_rows.groupby(group_keys, sort=False)["x"].median()
            GK_TRUST_THRESH = 30.0
            trusted = gk_med_x[gk_med_x.abs() >= GK_TRUST_THRESH]
            defend_x.loc[trusted.index] = trusted

    flip_groups = set(defend_x[defend_x > 0].index.tolist())
    flip_mask = out[group_keys].apply(tuple, axis=1).isin(flip_groups)

    out["x_norm"] = out["x"].copy()
    out["y_norm"] = out["y"].copy()
    out.loc[flip_mask, "x_norm"] = -out.loc[flip_mask, "x_norm"]
    out.loc[flip_mask, "y_norm"] = -out.loc[flip_mask, "y_norm"]

    in_bounds = (
        (out["x_norm"] >= X_MIN) & (out["x_norm"] <= X_MAX) &
        (out["y_norm"] >= Y_MIN) & (out["y_norm"] <= Y_MAX)
    )
    out = out[in_bounds].copy()
    return out


def main() -> None:
    """
    Sanity check for attack-normalization (GK sanity-check variant):
      - Load one match parquet.
      - Apply attack-normalization so each team attacks left-to-right (+X).
      - Choose a random position/role.
      - Plot a heatmap of all normalized (x_norm, y_norm) points for that position.
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

    norm = attack_normalize_like_gk_sanity(df)

    # Prefer Left Back for debugging; fall back to a frequent random role if missing
    roles = norm[role_col].astype(str)
    if (roles == "Attacking Midfield").any():
        chosen_role = "Attacking Midfield"
    else:
        role_counts = roles.value_counts()
        role_counts = role_counts[role_counts >= 500] if (role_counts >= 500).any() else role_counts
        chosen_role = role_counts.sample(n=1, random_state=42).index[0]

    sub = norm[roles == str(chosen_role)].dropna(subset=["x_norm", "y_norm", team_col])
    if sub.empty:
        raise ValueError(f"No rows found for chosen role={chosen_role!r} after normalization.")

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#1e1e1e",
        line_color="#ffffff",
        linewidth=1.5,
        goal_type="box",
    )

    # Bin edges for plotting space (0..105, 0..68)
    x_edges = np.linspace(0.0, 105.0, 75)
    y_edges = np.linspace(0.0, 68.0, 50)

    teams = sorted(sub[team_col].unique().tolist())
    if len(teams) > 2:
        teams = teams[:2]

    fig, axes = plt.subplots(1, len(teams), figsize=(12, 6.2), constrained_layout=True)
    if len(teams) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1e1e1e")

    # Precompute log-heatmaps per team so we can share a common scale
    log_maps = []
    for t in teams:
        st = sub[sub[team_col] == t]
        x_plot = st["x_norm"].to_numpy() + 52.5
        y_plot = st["y_norm"].to_numpy() + 34.0
        h, _, _ = np.histogram2d(x_plot, y_plot, bins=[x_edges, y_edges])
        log_maps.append(np.log1p(h).T)  # (y,x)

    vmax = max(float(m.max()) for m in log_maps) if log_maps else 1.0

    im = None
    for ax, t, log_h in zip(axes, teams, log_maps):
        pitch.draw(ax=ax)
        ax.set_facecolor("#1e1e1e")
        im = ax.imshow(
            log_h,
            extent=[0.0, 105.0, 0.0, 68.0],
            origin="lower",
            cmap="magma",
            alpha=0.9,
            interpolation="bilinear",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{team_col}={t}", color="white", fontsize=10, pad=8, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("log(1 + count)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    match_id = (
        int(norm["match_id"].dropna().iloc[0])
        if "match_id" in norm.columns and norm["match_id"].notna().any()
        else match_path.stem
    )
    fig.suptitle(
        f"Attack-normalized location heatmap (defending-side via trusted GK median-x else team q10)\n"
        f"match={match_id} | {role_col}={chosen_role} | split by {team_col}",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    out_path = PROJECT_ROOT / "position_attacknorm_heatmap_by_team.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Used file: {match_path}")
    print(f"Chosen {role_col}: {chosen_role}")
    for t in teams:
        print(f"Rows plotted for {team_col}={t}: {int((sub[team_col] == t).sum())}")


if __name__ == "__main__":
    main()


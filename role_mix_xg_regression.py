from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"
AUTOENCODER_DIR = PROJECT_ROOT / "data" / "outputs" / "autoencoder"
OUT_CLUSTER_DIR = PROJECT_ROOT / "data" / "outputs" / "clusters"
OUT_REG_DIR = PROJECT_ROOT / "data" / "outputs" / "regression"
OUT_REG_DIR.mkdir(parents=True, exist_ok=True)

TEAM_XG_CACHE = OUT_REG_DIR / "team_xg_per_match.csv"
TEAM_OUTFIELD_ROLE_CACHE = OUT_REG_DIR / "team_outfield_role_shares.csv"
TEAM_XG_DEDUP_CACHE = OUT_REG_DIR / "team_xg_per_match_dedup_by_event_id.csv"


def _extract_match_id_from_filename(path: Path) -> int | None:
    m = re.search(r"(\\d+)", path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_team_role_mixtures() -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Rebuild team-level role mixtures from player-level cluster probs and players_df mapping.
    Returns:
      team_probs: index=team_id, columns=prob_cluster_0..K
      label_map: team_id -> anonymised 'Team i' label
    """
    clusters_csv = AUTOENCODER_DIR / "autoencoder_gmm_clusters.csv"
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Missing autoencoder clustering CSV: {clusters_csv}")

    clusters_df = pd.read_csv(clusters_csv)
    if "player_id" not in clusters_df.columns:
        raise ValueError("autoencoder_gmm_clusters.csv must contain 'player_id'.")

    prob_cols = [c for c in clusters_df.columns if c.startswith("prob_cluster_")]
    if not prob_cols:
        raise ValueError("autoencoder_gmm_clusters.csv must contain prob_cluster_* columns.")

    with open(CREDS_FILE) as f:
        cfg = json.load(f)

    data_root = Path(cfg["data_folder_path"])
    players_csv = data_root / "skillcorner" / "players_df.csv"
    if not players_csv.exists():
        raise FileNotFoundError(f"players_df.csv not found at expected location: {players_csv}")

    players_df = pd.read_csv(players_csv)
    cols = [c for c in players_df.columns if c in {"player_id", "team_id"}]
    if "player_id" not in cols or "team_id" not in cols:
        raise ValueError("players_df.csv must contain 'player_id' and 'team_id' columns.")

    players_df = players_df[cols].copy()

    merged = clusters_df.merge(players_df, on="player_id", how="inner")
    if merged.empty:
        raise ValueError("Merge of clusters with players_df is empty; check player_id consistency.")

    team_probs = merged.groupby("team_id")[prob_cols].mean()

    # Build anonymised labels (Team 1, Team 2, ...)
    team_ids = team_probs.index.to_list()
    label_map = {int(tid): f"Team {i+1}" for i, tid in enumerate(team_ids)}

    return team_probs, label_map


def compute_team_xg_per_match() -> pd.DataFrame:
    """
    Iterate over merged per-match parquet files and compute average xG per match for each team_id.
    Returns a dataframe with columns: team_id, xg_per_match
    """
    # Previous versions of this script accidentally summed replicated tracking rows,
    # massively inflating xG. We now de-duplicate by `event_id` before aggregating.
    if TEAM_XG_DEDUP_CACHE.exists():
        cached = pd.read_csv(TEAM_XG_DEDUP_CACHE)
        if "team_id" in cached.columns and "xg_per_match" in cached.columns:
            return cached

    if not CREDS_FILE.exists():
        raise FileNotFoundError(f"Missing creds file: {CREDS_FILE}")

    with open(CREDS_FILE) as f:
        cfg = json.load(f)

    merged_dir = Path(cfg["merged_parquets_folder_path"])
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged matches directory not found: {merged_dir}")

    files = sorted(merged_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in merged directory: {merged_dir}")

    import pyarrow.parquet as pq

    team_match_xg: Dict[Tuple[int, int], float] = {}

    for i, p in enumerate(files, start=1):
        if i == 1 or i % 25 == 0:
            print(f"[xG] Processing match file {i}/{len(files)}: {p.name}")

        try:
            cols_available = set(pq.ParquetFile(p).schema.names)
        except Exception:
            continue

        base_cols: List[str] = []
        if "match_id" in cols_available:
            base_cols.append("match_id")
        # `players_df.csv` uses SkillCorner-style team ids.
        # In merged parquets, both `team_id` and `team` exist; empirically `team`
        # matches SkillCorner ids (while `team_id` is a different coding).
        team_col = "team" if "team" in cols_available else ("team_id" if "team_id" in cols_available else None)
        if team_col is None:
            continue
        base_cols.append(team_col)
        if "shot_statsbomb_xg" not in cols_available:
            continue
        base_cols.append("shot_statsbomb_xg")
        if "event_id" not in cols_available:
            continue
        base_cols.append("event_id")

        try:
            df = pd.read_parquet(p, columns=base_cols, engine="pyarrow")
        except Exception:
            continue

        if "match_id" in df.columns:
            mid = pd.to_numeric(df["match_id"], errors="coerce")
            if mid.notna().any():
                df["match_id"] = mid.astype("Int64")
            else:
                inferred = _extract_match_id_from_filename(p)
                if inferred is None:
                    continue
                df["match_id"] = inferred
        else:
            inferred = _extract_match_id_from_filename(p)
            if inferred is None:
                continue
            df["match_id"] = inferred

        df["shot_statsbomb_xg"] = pd.to_numeric(df["shot_statsbomb_xg"], errors="coerce")
        df["event_id"] = df["event_id"].astype(str)

        # Keep only rows that actually correspond to shots with xG.
        # Non-shot rows typically have NaN xG.
        df_valid = df.dropna(subset=["shot_statsbomb_xg"])
        if df_valid.empty:
            continue

        # Ensure aggregation keys are numeric so `int()` conversion won't silently fail.
        df_valid[team_col] = pd.to_numeric(df_valid[team_col], errors="coerce")
        df_valid = df_valid.dropna(subset=[team_col, "match_id", "event_id"])

        # De-duplicate replicated event rows: count each unique (match_id, team, event_id) once.
        # Use max() in case of tiny floating drift across replicated rows.
        event_xg = (
            df_valid.groupby(["match_id", team_col, "event_id"], sort=False)["shot_statsbomb_xg"].max()
        )
        team_xg = event_xg.groupby(level=[0, 1], sort=False).sum()

        for (mid, tid_raw), val in team_xg.items():
            try:
                tid = int(tid_raw)
                key_mid = int(mid)
            except (TypeError, ValueError):
                continue
            key = (tid, key_mid)
            team_match_xg[key] = team_match_xg.get(key, 0.0) + float(val)

    # Aggregate per team: mean xG per match
    per_team: Dict[int, List[float]] = {}
    for (tid, _mid), val in team_match_xg.items():
        per_team.setdefault(tid, []).append(val)

    rows = []
    for tid, vals in per_team.items():
        if not vals:
            continue
        rows.append({"team_id": tid, "xg_per_match": float(np.mean(vals))})

    team_xg = pd.DataFrame(rows)
    team_xg.to_csv(TEAM_XG_DEDUP_CACHE, index=False)
    # Also overwrite the canonical file name so downstream notebooks/scripts pick up the fix.
    team_xg.to_csv(TEAM_XG_CACHE, index=False)
    return team_xg


def run_ols(X: pd.DataFrame, y: pd.Series, base_cluster: int, out_name: str) -> None:
    """
    OLS via numpy.linalg.lstsq (with intercept), plus:
      - standard errors
      - t-stats
      - two-sided p-values
    based on classic OLS assumptions.
    """
    cols = list(X.columns)
    X_mat = X.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)
    n = len(y_vec)
    p = X_mat.shape[1]  # number of predictors (excluding intercept)
    k = p + 1  # parameters including intercept

    # Add intercept
    X_design = np.column_stack([np.ones(len(X_mat)), X_mat])

    beta, residuals, rank, s = np.linalg.lstsq(X_design, y_vec, rcond=None)

    y_hat = X_design @ beta
    ss_res = np.sum((y_vec - y_hat) ** 2)
    ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"Not enough degrees of freedom to compute p-values: n={n}, k={k}")

    mse = ss_res / df_resid
    # Use pseudo-inverse for numerical stability.
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    se = np.sqrt(np.diag(mse * xtx_inv))

    t_stats = beta / se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=df_resid)

    lines: List[str] = []
    lines.append(f"=== OLS regression: base cluster {base_cluster} (GK removed, outfield renormalised) ===")
    lines.append(f"n_samples = {len(y_vec)}, n_predictors = {len(cols)}")
    lines.append(f"R^2 = {r2:.4f}")
    lines.append(f"df_resid = {df_resid}")
    lines.append("")
    lines.append("Coefficients (including intercept) with SE / t / p-value:")
    lines.append(f"  intercept: coef={beta[0]: .6f}, se={se[0]: .6f}, t={t_stats[0]: .3f}, p={p_values[0]:.6g}")
    # Deterministic coefficient line printing by index.
    for j, name in enumerate(cols, start=1):
        lines.append(
            f"  {name}: coef={beta[j]: .6f}, se={se[j]: .6f}, t={t_stats[j]: .3f}, p={p_values[j]:.6g}"
        )

    summary = "\n".join(lines)
    print(summary)

    out_path = OUT_REG_DIR / out_name
    with open(out_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved regression summary to: {out_path}")


def main() -> None:
    # 1) Team role mixtures
    team_probs, label_map = load_team_role_mixtures()

    # 2) Remove GK cluster (assumed prob_cluster_0) and renormalise outfield roles
    outfield_cols = [c for c in team_probs.columns if c.startswith("prob_cluster_") and not c.endswith("_0")]
    if f"prob_cluster_0" in team_probs.columns:
        gk = team_probs["prob_cluster_0"]
        outfield_sum = team_probs[outfield_cols].sum(axis=1)
    else:
        # Fall back: assume the lowest-index cluster is GK
        sorted_cols = sorted([c for c in team_probs.columns if c.startswith("prob_cluster_")])
        gk_col = sorted_cols[0]
        outfield_cols = [c for c in sorted_cols if c != gk_col]
        gk = team_probs[gk_col]
        outfield_sum = team_probs[outfield_cols].sum(axis=1)

    # Avoid divide-by-zero; if outfield_sum is zero, drop that team
    valid_mask = outfield_sum > 0
    team_probs_out = team_probs.loc[valid_mask, outfield_cols].div(outfield_sum[valid_mask], axis=0)

    # Cache predictors table so repeated runs don't redo merges/groupbys.
    team_probs_out.to_csv(TEAM_OUTFIELD_ROLE_CACHE)

    # 3) Team xG per match
    team_xg = compute_team_xg_per_match()
    team_xg = team_xg.set_index("team_id")

    # 4) Align on team_id and build modelling table
    common_ids = team_probs_out.index.intersection(team_xg.index)
    if common_ids.empty:
        raise ValueError("No overlap between teams in role mixtures and xG per match.")

    X_full = team_probs_out.loc[common_ids].copy()
    y_full = team_xg.loc[common_ids, "xg_per_match"].copy()

    print(f"\nUsing {len(common_ids)} teams with both role mixtures and xG data.")

    # 5) Two OLS runs with different bases
    # Determine which column corresponds to cluster 1 / 5 (assuming naming prob_cluster_k)
    # Extract cluster indices from column names
    cluster_indices = {
        int(c.replace("prob_cluster_", "")): c for c in X_full.columns if c.startswith("prob_cluster_")
    }

    for base in (1, 5):
        if base not in cluster_indices:
            print(f"Skipping base cluster {base}: no column found.")
            continue

        base_col = cluster_indices[base]
        X_base = X_full.drop(columns=[base_col])
        out_name = f"ols_base_cluster{base}.txt"
        run_ols(X_base, y_full, base_cluster=base, out_name=out_name)


if __name__ == "__main__":
    main()


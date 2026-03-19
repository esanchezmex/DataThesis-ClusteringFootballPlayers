"""
Cluster statistics table (per-90 + pass length, etc.)
----------------------------------------------------

Builds a crosstab-style table of aggregated per-cluster football metrics such as:
  - shots per 90
  - crosses per 90
  - passes per 90
  - carries/dribbles per 90
  - xG per 90 (StatsBomb xG if available)
  - average pass length

Minutes played is estimated per player by summing, over matches, the span of
tracking timestamps for that player: (max_t - min_t) / 60.
This is robust for full-match players and good enough for per-90 scaling.

Inputs:
  - data/outputs/autoencoder/ml_ready_features_optimal.csv (latent vectors)
  - creds/gdrive_folder.json -> final_data/*.parquet (tracking + events)

Outputs:
  - Prints a cluster x metric table to stdout (similar spirit to existing crosstabs)
  - Saves CSV: data/outputs/clusters/cluster_stats_table.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import pyarrow.parquet as pq
import ast


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"
AUTOENCODER_CSV = PROJECT_ROOT / "data" / "outputs" / "autoencoder" / "ml_ready_features_optimal.csv"
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def resolve_final_data_dir() -> Path:
    with open(CREDS_FILE, "r") as f:
        cfg = json.load(f)
    final_data_dir = Path(cfg["final_data"])
    if not final_data_dir.exists():
        raise FileNotFoundError(f"final_data directory not found: {final_data_dir}")
    return final_data_dir


def ensure_primary_cluster(clusters_df: pd.DataFrame) -> pd.DataFrame:
    if "primary_cluster" in clusters_df.columns:
        return clusters_df

    latent_cols = [c for c in clusters_df.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError("ml_ready_features_optimal.csv must contain latent_* columns to infer clusters.")

    Z = clusters_df[latent_cols].to_numpy(dtype=np.float64)

    best_bic = float("inf")
    best_gmm: Optional[GaussianMixture] = None
    best_n: Optional[int] = None

    for n in range(3, 11):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=3,
            random_state=SEED,
        )
        gmm.fit(Z)
        bic = gmm.bic(Z)
        if bic < best_bic:
            best_bic, best_gmm, best_n = bic, gmm, n

    out = clusters_df[["player_id"]].copy()
    out["primary_cluster"] = best_gmm.predict(Z)
    out.attrs["n_components"] = best_n
    return out


def _truthy_series(s: pd.Series) -> pd.Series:
    """
    Convert heterogeneous "boolean-like" series to boolean.
    Handles True/False, 1/0, "True"/"False", "t"/"f", etc.
    """
    if s.dtype == bool:
        return s
    x = s.astype(str).str.lower()
    return x.isin(["true", "1", "t", "yes", "y"])


def _get_time_seconds(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Prefer `timestamp` (seconds). Fall back to minute/second.
    Returns a Series of seconds or None if unavailable.
    """
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce")
        if t.notna().any():
            return t

    minute_col = "minute" if "minute" in df.columns else ("event_minute" if "event_minute" in df.columns else None)
    second_col = "second" if "second" in df.columns else ("seconds" if "seconds" in df.columns else None)
    if minute_col is None or second_col is None:
        return None

    m = pd.to_numeric(df[minute_col], errors="coerce")
    s = pd.to_numeric(df[second_col], errors="coerce")
    t = m * 60.0 + s
    return t


def accumulate_player_minutes(final_data_dir: Path) -> Dict[int, float]:
    minutes: Dict[int, float] = {}

    files = sorted(final_data_dir.glob("*.parquet"))
    for i, p in enumerate(files, start=1):
        if i == 1 or i % 25 == 0:
            print(f"  minutes: [{i}/{len(files)}] {p.name}")

        cols_available = set(pq.ParquetFile(p).schema.names)
        cols = [c for c in ["player_id", "timestamp", "minute", "event_minute", "second", "seconds"] if c in cols_available]
        if "player_id" not in cols:
            continue

        df = pd.read_parquet(p, columns=cols)
        df = df.dropna(subset=["player_id"])
        df["player_id"] = df["player_id"].astype("int64")

        t = _get_time_seconds(df)
        if t is None:
            continue
        df = df.assign(_t=t).dropna(subset=["_t"])

        span = df.groupby("player_id")["_t"].agg(["min", "max"])
        # ignore pathological cases
        dt = (span["max"] - span["min"]).clip(lower=0.0)
        add_min = (dt / 60.0).to_dict()
        for pid, m in add_min.items():
            minutes[int(pid)] = minutes.get(int(pid), 0.0) + float(m)

    return minutes


def accumulate_event_stats(final_data_dir: Path) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, float], Dict[int, Tuple[float, int]]]:
    shots: Dict[int, int] = {}
    crosses: Dict[int, int] = {}
    passes: Dict[int, int] = {}
    carries: Dict[int, int] = {}
    xg: Dict[int, float] = {}
    pass_len_sum_cnt: Dict[int, Tuple[float, int]] = {}

    files = sorted(final_data_dir.glob("*.parquet"))
    for i, p in enumerate(files, start=1):
        if i == 1 or i % 25 == 0:
            print(f"  events:  [{i}/{len(files)}] {p.name}")

        cols_available = set(pq.ParquetFile(p).schema.names)
        base_cols = [c for c in [
            "match_id", "period", "team", "team_id", "role_name", "position",
            "event_type", "type", "player_id", "event_player_id",
            "event_id", "x", "y",
            "pass_cross", "pass_type", "pass_length",
            "shot_statsbomb_xg",
        ] if c in cols_available]
        # `event_location` is often stored as a nested/list column and may not appear in
        # pyarrow schema names, but pandas can still materialize it. Always request it.
        needed = list(dict.fromkeys(base_cols + ["event_location"]))

        try:
            df = pd.read_parquet(p, columns=needed)
        except Exception:
            continue

        # Need team/half info to align tracking coords to attack-normalized event space.
        team_col = "team" if "team" in df.columns else ("team_id" if "team_id" in df.columns else None)
        if team_col is None or "period" not in df.columns:
            continue
        role_col = "role_name" if "role_name" in df.columns else ("position" if "position" in df.columns else None)

        # Identify event type (prefer event_type, but fall back to type if event_type is mostly null)
        ev_col = "event_type" if "event_type" in df.columns else ("type" if "type" in df.columns else None)
        if ev_col is None:
            continue

        ev = df[ev_col].astype(str).str.lower()
        if ev.eq("nan").mean() > 0.95 and "type" in df.columns and ev_col != "type":
            ev = df["type"].astype(str).str.lower()

        pid_col = "player_id" if "player_id" in df.columns else ("event_player_id" if "event_player_id" in df.columns else None)
        if pid_col is None:
            continue

        if "event_id" not in df.columns:
            continue

        # --- Attack-normalize tracking coordinates to match StatsBomb event space ---
        # StatsBomb event_location is always in attacking direction (left->right).
        # We transform tracking x/y into the same frame (x_norm/y_norm) using team+half inference.
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        if df["x"].notna().sum() == 0 or df["y"].notna().sum() == 0:
            continue

        grp_keys = ["period", team_col]
        team_q10_x = df.groupby(grp_keys, sort=False)["x"].quantile(0.10)
        defend_x = team_q10_x.copy()

        if role_col is not None:
            gk_rows = df[df[role_col].astype(str).str.lower().eq("goalkeeper")]
            if not gk_rows.empty:
                gk_med_x = gk_rows.groupby(grp_keys, sort=False)["x"].median()
                trusted = gk_med_x[gk_med_x.abs() >= 30.0]
                defend_x.loc[trusted.index] = trusted

        flip_groups = set(defend_x[defend_x > 0].index.tolist())
        group_tuples = list(zip(df["period"].values, df[team_col].values))
        flip_mask = pd.Series(group_tuples, index=df.index).isin(flip_groups)

        df["x_norm"] = df["x"]
        df["y_norm"] = df["y"]
        df.loc[flip_mask, "x_norm"] = -df.loc[flip_mask, "x_norm"]
        df.loc[flip_mask, "y_norm"] = -df.loc[flip_mask, "y_norm"]
        # ------------------------------------------------------------------------

        pid_raw = pd.to_numeric(df[pid_col], errors="coerce")
        eid_raw = df["event_id"].astype(str)
        keep_ev = ev.isin(["shot", "pass", "carry", "dribble"])
        keep = keep_ev & pid_raw.notna() & eid_raw.notna() & ~eid_raw.str.lower().eq("nan")
        if not keep.any():
            continue

        # We need to attribute each event to the *actor* in tracking-id space.
        # In these merged tables, an event is replicated across many tracking rows (one per player),
        # so we choose the row whose (x,y) is closest to the event_location for that event_id.
        if "event_location" not in df.columns or "x" not in df.columns or "y" not in df.columns:
            continue

        sub = df.loc[keep, ["event_location", "x_norm", "y_norm"]].copy()
        sub["event_id"] = eid_raw[keep].values
        sub["ev"] = ev[keep].values
        sub["player_id"] = pid_raw[keep].astype("int64").values

        # Parse event_location into numeric x,y (handle list, tuple, or stringified list)
        def _parse_loc(v):
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                ex, ey = float(v[0]), float(v[1])
                # StatsBomb-style event space is typically x∈[0,120], y∈[0,80].
                # Convert to centered 105x68 meters: x_c = (x/120)*105 - 52.5, y_c = (y/80)*68 - 34.
                if 0.0 <= ex <= 120.0 and 0.0 <= ey <= 80.0:
                    ex = (ex / 120.0) * 105.0 - 52.5
                    ey = (ey / 80.0) * 68.0 - 34.0
                return ex, ey
            if isinstance(v, str):
                try:
                    vv = ast.literal_eval(v)
                    if isinstance(vv, (list, tuple, np.ndarray)) and len(vv) >= 2:
                        ex, ey = float(vv[0]), float(vv[1])
                        if 0.0 <= ex <= 120.0 and 0.0 <= ey <= 80.0:
                            ex = (ex / 120.0) * 105.0 - 52.5
                            ey = (ey / 80.0) * 68.0 - 34.0
                        return ex, ey
                except Exception:
                    return np.nan, np.nan
            return np.nan, np.nan

        loc_xy = sub["event_location"].apply(_parse_loc)
        sub["_ex"] = [a for a, _ in loc_xy]
        sub["_ey"] = [b for _, b in loc_xy]

        sub["x_norm"] = pd.to_numeric(sub["x_norm"], errors="coerce")
        sub["y_norm"] = pd.to_numeric(sub["y_norm"], errors="coerce")
        sub = sub.dropna(subset=["_ex", "_ey", "x_norm", "y_norm"])
        if sub.empty:
            continue

        sub["_d2"] = (sub["x_norm"] - sub["_ex"]) ** 2 + (sub["y_norm"] - sub["_ey"]) ** 2

        # Pick the closest tracking row per event_id
        idx = sub.groupby("event_id")["_d2"].idxmin()
        chosen = sub.loc[idx, ["event_id", "player_id", "ev"]].copy()

        # Pull event attributes from the *chosen original rows*.
        # `idx` are original row labels in the parquet dataframe (RangeIndex), so we can use .loc safely.
        chosen_rows = sub.loc[idx].copy()
        ev_df = chosen_rows[["event_id", "player_id", "ev"]].copy()
        if "pass_cross" in df.columns:
            ev_df["pass_cross"] = _truthy_series(df.loc[idx, "pass_cross"]).values
        if "pass_type" in df.columns:
            ev_df["pass_type"] = df.loc[idx, "pass_type"].astype(str).str.lower().values
        if "pass_length" in df.columns:
            ev_df["pass_length"] = pd.to_numeric(df.loc[idx, "pass_length"], errors="coerce").values
        if "shot_statsbomb_xg" in df.columns:
            ev_df["shot_statsbomb_xg"] = pd.to_numeric(df.loc[idx, "shot_statsbomb_xg"], errors="coerce").values

        # Shots
        shot_df = ev_df[ev_df["ev"] == "shot"]
        if not shot_df.empty:
            vc = shot_df["player_id"].value_counts()
            for k, v in vc.items():
                shots[int(k)] = shots.get(int(k), 0) + int(v)
            if "shot_statsbomb_xg" in shot_df.columns:
                sxg = shot_df.groupby("player_id")["shot_statsbomb_xg"].sum(min_count=1)
                for k, v in sxg.items():
                    if pd.notna(v):
                        xg[int(k)] = xg.get(int(k), 0.0) + float(v)

        # Passes + crosses + pass length
        pass_df = ev_df[ev_df["ev"] == "pass"]
        if not pass_df.empty:
            vc = pass_df["player_id"].value_counts()
            for k, v in vc.items():
                passes[int(k)] = passes.get(int(k), 0) + int(v)

            is_cross = None
            if "pass_cross" in pass_df.columns:
                is_cross = pass_df["pass_cross"].fillna(False).astype(bool)
            if "pass_type" in pass_df.columns:
                has_cross = pass_df["pass_type"].fillna("").str.contains("cross", na=False)
                is_cross = has_cross if is_cross is None else (is_cross | has_cross)
            if is_cross is not None:
                vcx = pass_df.loc[is_cross, "player_id"].value_counts()
                for k, v in vcx.items():
                    crosses[int(k)] = crosses.get(int(k), 0) + int(v)

            if "pass_length" in pass_df.columns:
                pl = pass_df.dropna(subset=["pass_length"])
                if not pl.empty:
                    sums = pl.groupby("player_id")["pass_length"].agg(["sum", "count"])
                    for k, row in sums.iterrows():
                        prev_sum, prev_cnt = pass_len_sum_cnt.get(int(k), (0.0, 0))
                        pass_len_sum_cnt[int(k)] = (prev_sum + float(row["sum"]), prev_cnt + int(row["count"]))

        # Carries/dribbles
        carry_df = ev_df[ev_df["ev"].isin(["carry", "dribble"])]
        if not carry_df.empty:
            vc = carry_df["player_id"].value_counts()
            for k, v in vc.items():
                carries[int(k)] = carries.get(int(k), 0) + int(v)

    return shots, crosses, passes, carries, xg, pass_len_sum_cnt


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_colwidth", None)

    if not AUTOENCODER_CSV.exists():
        raise FileNotFoundError(f"Missing: {AUTOENCODER_CSV}")

    clusters_raw = pd.read_csv(AUTOENCODER_CSV, usecols=lambda c: c == "player_id" or c.startswith("latent_"))
    clusters = ensure_primary_cluster(clusters_raw)
    n_components = clusters.attrs.get("n_components", None)

    final_data_dir = resolve_final_data_dir()

    print("Accumulating minutes played per player (from timestamps) …")
    minutes = accumulate_player_minutes(final_data_dir)
    print(f"Minutes computed for {len(minutes)} players.")

    print("Accumulating event stats per player (shots, crosses, passes, carries, xG, pass length) …")
    shots, crosses, passes, carries, xg, pass_len_sum_cnt = accumulate_event_stats(final_data_dir)

    stats_df = pd.DataFrame({"player_id": clusters["player_id"].astype("int64")})
    stats_df["minutes"] = stats_df["player_id"].map(minutes).fillna(0.0)
    stats_df["shots"] = stats_df["player_id"].map(shots).fillna(0).astype(int)
    stats_df["crosses"] = stats_df["player_id"].map(crosses).fillna(0).astype(int)
    stats_df["passes_evt"] = stats_df["player_id"].map(passes).fillna(0).astype(int)
    stats_df["carries_evt"] = stats_df["player_id"].map(carries).fillna(0).astype(int)
    stats_df["xg"] = stats_df["player_id"].map(xg).fillna(0.0)

    def _avg_pass_len(pid: int) -> float:
        s, c = pass_len_sum_cnt.get(int(pid), (0.0, 0))
        return float(s) / float(c) if c > 0 else float("nan")

    stats_df["avg_pass_length"] = stats_df["player_id"].apply(_avg_pass_len)

    merged = clusters.merge(stats_df, on="player_id", how="left")

    # Cluster-level totals first
    agg_totals = merged.groupby("primary_cluster").agg(
        n_players=("player_id", "count"),
        minutes_total=("minutes", "sum"),
        shots_total=("shots", "sum"),
        crosses_total=("crosses", "sum"),
        passes_total=("passes_evt", "sum"),
        carries_total=("carries_evt", "sum"),
        xg_total=("xg", "sum"),
    ).sort_index()

    # Per-90 scaling at cluster level: (total / minutes_total) * 90
    denom = (agg_totals["minutes_total"] / 90.0).replace(0.0, np.nan)
    agg = pd.DataFrame(index=agg_totals.index)
    agg["n_players"] = agg_totals["n_players"]
    agg["minutes_total"] = agg_totals["minutes_total"]
    agg["shots_per90"] = agg_totals["shots_total"] / denom
    agg["crosses_per90"] = agg_totals["crosses_total"] / denom
    agg["passes_per90"] = agg_totals["passes_total"] / denom
    agg["carries_per90"] = agg_totals["carries_total"] / denom
    agg["xg_per90"] = agg_totals["xg_total"] / denom

    # Cluster-level average pass length: total pass length / total passes with length
    # Use the per-player avg_pass_length only as fallback if needed.
    pass_len_sum = merged["player_id"].map(lambda pid: pass_len_sum_cnt.get(int(pid), (0.0, 0))[0])
    pass_len_cnt = merged["player_id"].map(lambda pid: pass_len_sum_cnt.get(int(pid), (0, 0))[1])
    merged = merged.assign(_pl_sum=pass_len_sum.values, _pl_cnt=pass_len_cnt.values)
    pl_agg = merged.groupby("primary_cluster")[["_pl_sum", "_pl_cnt"]].sum()
    agg["avg_pass_length"] = pl_agg["_pl_sum"] / pl_agg["_pl_cnt"].replace(0, np.nan)

    # Pretty formatting
    pretty = agg.copy()
    for c in ["minutes_total"]:
        pretty[c] = pretty[c].round(0).astype("Int64")
    for c in ["shots_per90", "crosses_per90", "passes_per90", "carries_per90", "xg_per90", "avg_pass_length"]:
        pretty[c] = pretty[c].astype(float).round(2)

    header = []
    if n_components is not None:
        header.append(f"BIC-selected n_components = {n_components} (diag cov, reg_covar=1e-4)")
    header.append("Cluster-level stats (cluster totals / cluster minutes * 90; minutes from timestamp spans)")
    print("\n" + "\n".join(header) + "\n")
    print(pretty.to_string())

    out_csv = OUT_DIR / "cluster_stats_table.csv"
    pretty.reset_index().rename(columns={"primary_cluster": "cluster"}).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()


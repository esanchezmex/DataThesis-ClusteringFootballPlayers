import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parent
CREDS_FILE = PROJECT_ROOT / "creds" / "gdrive_folder.json"


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_colwidth", None)

    clusters_path = PROJECT_ROOT / "data" / "outputs" / "autoencoder" / "ml_ready_features_optimal.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Missing: {clusters_path}")

    clusters = pd.read_csv(clusters_path)
    if "player_id" not in clusters.columns:
        raise ValueError("ml_ready_features_optimal.csv must contain player_id")

    latent_cols = [c for c in clusters.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError("ml_ready_features_optimal.csv must contain latent_* columns for Option B.")

    Z = clusters[latent_cols].to_numpy(dtype=np.float64)

    # BIC sweep to choose n_components in [3, 10]
    best_bic = float("inf")
    best_gmm = None
    best_n = None
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
            best_bic, best_gmm, best_n = bic, gmm, n

    clusters = clusters[["player_id"]].copy()
    clusters["primary_cluster"] = best_gmm.predict(Z)

    # Dominant role_name per player from final_data match parquets
    with open(CREDS_FILE) as f:
        cfg = json.load(f)
    final_data_dir = Path(cfg["final_data"])
    if not final_data_dir.exists():
        raise FileNotFoundError(f"final_data directory not found: {final_data_dir}")

    acc = None  # Series indexed by (player_id, role_name)
    for p in sorted(final_data_dir.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["player_id", "role_name"]).dropna(subset=["player_id", "role_name"])
        df["player_id"] = df["player_id"].astype("int64")
        vc = df.value_counts(subset=["player_id", "role_name"])
        acc = vc if acc is None else acc.add(vc, fill_value=0)

    acc = acc.astype("int64")
    role_map = acc.groupby(level=0).idxmax().reset_index()
    role_map.columns = ["player_id", "position"]

    merged = clusters.merge(role_map, on="player_id", how="left")
    merged["position"] = merged["position"].fillna("Unknown")

    ct = pd.crosstab(merged["primary_cluster"], merged["position"]).sort_index()

    print(f"\nBIC-selected n_components = {best_n} (diag cov, reg_covar=1e-4)")
    print("Cluster x Tactical Role counts (Autoencoder latent → GMM)\n")
    print(ct.to_string())


if __name__ == "__main__":
    main()


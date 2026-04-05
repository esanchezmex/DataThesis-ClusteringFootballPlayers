from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA

from train_spatial_autoencoder import (
    OUTPUT_DIR,
    SpatialAutoencoder,
    extract_latent,
    get_device,
    load_and_normalize,
    load_player_roles,
    resolve_paths,
)


def main() -> None:
    # Same latent pipeline as the t-SNE figure
    profiles_pkl, final_data_dir = resolve_paths()
    tensors_array, _, player_ids = load_and_normalize(profiles_pkl)

    optimal_dim = 16
    device = get_device()
    model = SpatialAutoencoder(latent_dim=optimal_dim).to(device)
    weights_path = OUTPUT_DIR / f"weights_dim_{optimal_dim}.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    Z = extract_latent(model, tensors_array, device)

    # PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    Z2 = pca.fit_transform(Z)

    role_map = load_player_roles(final_data_dir)
    plot_df = pd.DataFrame(
        {"pc1": Z2[:, 0], "pc2": Z2[:, 1], "player_id": player_ids}
    )
    plot_df["role"] = plot_df["player_id"].map(role_map).fillna("Unknown")

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    palette = sns.color_palette("tab20", n_colors=plot_df["role"].nunique())
    role_order = sorted(plot_df["role"].unique())
    sns.scatterplot(
        data=plot_df,
        x="pc1",
        y="pc2",
        hue="role",
        hue_order=role_order,
        palette=palette,
        alpha=0.80,
        s=55,
        linewidth=0.3,
        ax=ax,
    )

    legend = ax.legend(
        title="Tactical Role",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10,
        title_fontsize=10,
        facecolor="white",
        labelcolor="black",
    )
    legend.get_title().set_color("black")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.tick_params(colors="black", labelsize=10)
    ax.set_xlabel("Principal Component 1", color="black", fontsize=12)
    ax.set_ylabel("Principal Component 2", color="black", fontsize=12)
    ax.set_title(
        "PCA of Optimal Latent Space (2 Components)\nColoured by Actual Tactical Role (from data provider)",
        color="black",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    print(
        f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.4f}, "
        f"Total={pca.explained_variance_ratio_[:2].sum():.4f}"
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "pca_quick_preview.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


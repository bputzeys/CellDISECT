import scvi

scvi.settings.seed = 0
import scanpy as sc
import anndata as ad
import torch
import gc

torch.set_float32_matmul_precision("medium")
import warnings

warnings.simplefilter("ignore", UserWarning)
import scanpy as sc

from celldisect import CellDISECT

adata = sc.read_h5ad("PATH/TO/DATA.h5ad")  # Write the path to your h5ad file
adata = adata[adata.X.sum(1) != 0].copy()

# Covariate keys from adata.obs to be observed in the model, add them here
cats = [
    "cat1",
    "cat2",
    "cat3",
    # ...
]
cell_type_included = (
    False  # Set to True if you have provided a cell type annotation in the cats list
)
if not cell_type_included:
    adata.obs["_cluster"] = (
        "0"  # Dummy obs for inference (not-training) time, to avoid computing neighbors and clusters again in setup_anndata | AVOID ADDING BEFORE TRAINING
    )

module_name = "PROJECT_NAME"
pre_path = f"/PATH/TO/SAVE/YOUR/MODEL/{module_name}"

model_name = "MODEL_NAME"
## Example:
# model_name = 'pretrainAE_0_maxEpochs_1000_reconW_20_cfWeight_0.8_beta_0.003_clf_0.05_adv_0.014_advp_5_n_cf_1_lr_0.003_wd_5e-05_new_cf_True_dropout_0.1_n_hidden_128_n_latent_32_n_layers_2_batch_size_256_cellTypeNotIncluded'

model = CellDISECT.load(f"{pre_path}/{model_name}", adata=adata)

# Get the latent representations
print(f"Getting the latent 0...")
# Z_0
adata.obsm[f"CellDISECT_Z_0"] = model.get_latent_representation(
    nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False
)

for i in range(len(cats)):
    print(f"Getting the latent {i+1} / {len(cats)}...")
    null_idx = [s for s in range(len(cats)) if s != i]
    label = cats[i]
    # Z_i
    adata.obsm[f"CellDISECT_Z_{label}"] = model.get_latent_representation(
        nullify_cat_covs_indices=null_idx, nullify_shared=True
    )
    # Z_{-i}
    adata.obsm[f"CellDISECT_Z_not_{label}"] = model.get_latent_representation(
        nullify_cat_covs_indices=[i], nullify_shared=False
    )

# Z_{0+Z_i} = Z_0 + Z_i | Optional
# This represents the combination of latents with Z_0.
# Here, Z_i is a vector where all elements are zero except for the i-th section.
# Z_0 is [latent_0, 0, ..., 0] and Z_i is [0, ..., 0, latent_i, 0, ..., 0].
# When mixing two latents, it will look like [latent_0, 0, latent_i, 0, ..., 0].
for i in range(len(cats)):  # loop over all Z_i
    label = cats[i]
    latent_name = f"CellDISECT_Z_0+{label}"
    adata.obsm[latent_name] = (
        adata.obsm["CellDISECT_Z_0"].copy() + adata.obsm[f"CellDISECT_Z_{label}"].copy()
    )

# Compute neighbors and UMAPs for the latent representations (this might take a while, consider running it using RAPIDS scanpy with a GPU if data is large)
for i in range(len(cats) + 1):  # loop over all Z_i | Neighbors and UMAPs for Z_i
    if i == 0:
        latent_name = f"CellDISECT_Z_{i}"
    else:
        label = cats[i - 1]
        latent_name = f"CellDISECT_Z_{label}"

    latent = ad.AnnData(X=adata.obsm[f"{latent_name}"], obs=adata.obs)
    sc.pp.neighbors(adata=latent, use_rep="X")
    sc.tl.umap(adata=latent)

    adata.uns[f"{latent_name}_neighbors"] = latent.uns["neighbors"]
    adata.obsm[f"{latent_name}_umap"] = latent.obsm["X_umap"]
    gc.collect()

for i in range(
    len(cats)
):  # loop over all Z_i | Neighbors and UMAPs for Z_0+Z_i | Optional
    label = cats[i]
    latent_name = f"CellDISECT_Z_0+{label}"

    latent = ad.AnnData(X=adata.obsm[f"{latent_name}"], obs=adata.obs)
    sc.pp.neighbors(adata=latent, use_rep="X")
    sc.tl.umap(adata=latent)

    adata.uns[f"{latent_name}_neighbors"] = latent.uns["neighbors"]
    adata.obsm[f"{latent_name}_umap"] = latent.obsm["X_umap"]
    gc.collect()

# Save the adata with the latent representations and neighbors
adata.write(f"/PATH/TO/SAVE/LATENTS.h5ad")

# # Code for plotting the UMAPs in your notebook later
# # Plotting Z_i
# colors = cats
# # colors = cats + ['any_other_obs_key']

# for i in range(len(cats) + 1):  # loop over all Z_i
#     if i == 0:
#         latent_name = f'CellDISECT_Z_{i}'
#     else:
#         label = cats[i-1]
#         latent_name = f'CellDISECT_Z_{label}'


#     print(f"---UMAP for {latent_name}---")
#     sc.set_figure_params(figsize=(12, 8))
#     sc.pl.embedding(
#         adata,
#         f'{latent_name}_umap',
#         color=colors,
#         ncols=len(colors),
#         frameon=False,
#         # legend_loc=None, # Uncomment this line if you want to remove the legend
#         # wspace=0.2,
#     )

# # Plotting Z_0+Z_i | Optional
# colors = cats
# # colors = cats + ['any_other_obs_key']

# for i in range(len(cats)):  # loop over all Z_i
#     label = cats[i]
#     latent_name = f'CellDISECT_Z_0+{label}'

#     print(f"---UMAP for {latent_name}---")
#     sc.set_figure_params(figsize=(12, 8))
#     sc.pl.embedding(
#         adata,
#         f'{latent_name}_umap',
#         color=colors,
#         ncols=len(colors),
#         frameon=False,
#         # legend_loc=None, # Uncomment this line if you want to remove the legend
#         # wspace=0.2,
#     )

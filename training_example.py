import os
import shutil

import scvi
import scanpy as sc
import torch
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)

from dis2p import dis2pvi_cE as dvi

scvi.settings.seed = 42

adata = sc.read_h5ad('PATH/TO/DATA.h5ad') # Write the path to your h5ad file
adata = adata[adata.X.sum(1) != 0].copy()

# Covariate keys from adata.obs to be observed in the model, add them here
cats = [
    'cat1',
    'cat2',
    'cat3',
    # ...
    ]
cell_type_included = False # Set to True if you have provided a cell type annotation in the cats list

module_name = 'PROJECT_NAME'
pre_path = f'/PATH/TO/SAVE/YOUR/MODEL/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)

arch_dict = {'n_layers': 2,
 'n_hidden': 128,
 'n_latent_shared': 32,
 'n_latent_attribute': 32,
 'dropout_rate': 0.1,
 'weighted_classifier': False,
}
train_dict = {
 'max_epochs': 1000,
 'batch_size': 256,
 'recon_weight': 20,
 'cf_weight': 0.8,
 'beta': 0.003,
 'clf_weight': 0.05,
 'adv_clf_weight': 0.014,
 'adv_period': 5,
 'n_cf': 1,
 'early_stopping_patience': 6,
 'early_stopping': True,
 'save_best': True,
 'kappa_optimizer2': False,
 'n_epochs_pretrain_ae': 0,
}

plan_kwargs = {
 'lr': 0.003,
 'weight_decay': 0.00005,
 'new_cf_method': True,
 'lr_patience': 5,
 'lr_factor': 0.5,
 'lr_scheduler_metric': 'loss_validation',
 'n_epochs_kl_warmup': 10,
}

# specify a name for your model
model_name = (
    f'pretrainAE_{train_dict["n_epochs_pretrain_ae"]}_'
    f'maxEpochs_{train_dict["max_epochs"]}_'
    f'reconW_{train_dict["recon_weight"]}_'
    f'cfWeight_{train_dict["cf_weight"]}_'
    f'beta_{train_dict["beta"]}_'
    f'clf_{train_dict["clf_weight"]}_'
    f'adv_{train_dict["adv_clf_weight"]}_'
    f'advp_{train_dict["adv_period"]}_'
    f'n_cf_{train_dict["n_cf"]}_'
    f'lr_{plan_kwargs["lr"]}_'
    f'wd_{plan_kwargs["weight_decay"]}_'
    f'new_cf_{plan_kwargs["new_cf_method"]}_'
    f'dropout_{arch_dict["dropout_rate"]}_'
    f'n_hidden_{arch_dict["n_hidden"]}_'
    f'n_latent_{arch_dict["n_latent_shared"]}_'
    f'n_layers_{arch_dict["n_layers"]}_'
    f'batch_size_{train_dict["batch_size"]}_'
)
if cell_type_included:
    model_name = model_name + f'cellTypeIncluded'
else:
    model_name = model_name + f'cellTypeNotIncluded'

wandb_logger = WandbLogger(project=f"Dis2PVI_cE_{module_name}", name=model_name) # If you have a wandb account logged in. Very recommended for training monitoring, feel free to comment
train_dict['logger'] = wandb_logger # If you have a wandb account logged in. Very recommended for training monitoring, feel free to comment
wandb_logger.experiment.config.update({'train_dict': train_dict, 'arch_dict': arch_dict, 'plan_kwargs': plan_kwargs}) # If you have a wandb account logged in. Very recommended for training monitoring, feel free to comment

try: # Clean up the directory if it exists, overwrite the model
    shutil.rmtree(f"{pre_path}/{model_name}")
    print("Directory deleted successfully")
except OSError as e:
    print(f"Error deleting directory: {e}") 


dvi.Dis2pVI_cE.setup_anndata(
    adata,
    layer='counts',
    categorical_covariate_keys=cats,
    continuous_covariate_keys=[],
    add_cluster_covariate=not cell_type_included, # add_cluster_covariate if cell type is not included
)

# Use this to make random splits
model = dvi.Dis2pVI_cE(adata,
                       **arch_dict)
# Use this if you have pre-defined splits
# model = dvi.Dis2pVI_cE(adata,
#                        split_key=split_key,
#                        train_split=['train'],
#                        valid_split=['val'],
#                        test_split=['test'],
#                        **arch_dict)

model.train(**train_dict, plan_kwargs=plan_kwargs, )
model.save(f"{pre_path}/{model_name}", overwrite=True)
print(model_name)

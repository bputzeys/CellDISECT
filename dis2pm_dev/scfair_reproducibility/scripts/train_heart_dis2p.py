import sys
import argparse
import os
import scvi
scvi.settings.seed = 0
import scanpy as sc
import anndata as ad
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import csr_matrix
torch.set_float32_matmul_precision('medium')
import warnings
warnings.simplefilter("ignore", UserWarning)


# append paths to sys.path to import from many locations (eg evaluate and scib)
current_script_path = os.path.dirname(__file__)
fpath = os.path.join(current_script_path, "..")
sys.path.append(fpath)
fpath = os.path.join(current_script_path, "../..")
sys.path.append(fpath)

from scfair_reproducibility.evaluation.metrics import *
from scfair_reproducibility.evaluation.diffair_evaluate import *

# from dis2p.dis2pvi import Dis2pVI
from dis2p.dis2pvae import *
from dis2p.dis2pvi import *
from dis2p.ood import *
from dis2p.trainingplan import *
from dis2p.utils import *

# run with
# python scripts/train_heart_dis2p.py --CFwei 1 --classwei 1 --advclasswei 0.5 -e 400 -b 128 -be 1 -advPER 1 -n_cf 1


# ------------  READ INPUTS  -------------------
parser = argparse.ArgumentParser(
                    prog='dis2p',
                    description='train dis2p specifying data, hyperparameters, destinations',
                    epilog='Text at the bottom of help')

print('getting arguments...')

parser.add_argument('-e', '--epochs',  type=int, default=400,
                    help='an integer for the number of epochs')
parser.add_argument('-b', '--batch', type=int, default=128,
                    help='an integer for the batch size')
parser.add_argument('-cfw', '--CFwei', type=float, default=1.0,
                    help='a float for the counterfactual weight')
parser.add_argument('-be', '--beta', type=float, default=1,
                    help='a float for the KL weight')
parser.add_argument('-CLFw', '--classwei', type=float, default=50.0,
                    help='a float for the weight of classification loss term')
parser.add_argument('-advCLFw', '--advclasswei', type=float, default=10.0,
                    help='a float for the weight of adversarial classification loss term')
parser.add_argument('-advPER', '--advperiod', type=int, default=1,
                    help='an integer for the adversarial period')
parser.add_argument('-n_cf', '--n_cf', type=int, default=1,
                    help='number of random permutations for the n VAEs per batch')

args = parser.parse_args()

print(args)
epochs = args.epochs
batch_size = args.batch
cf_weight = args.CFwei
beta = args.beta
clf_weight = args.classwei
adv_clf_weight = args.advclasswei
adv_period = args.advperiod
n_cf = args.n_cf

# ------------  PREPARE DATASETS AND DIRECTORIES --------------
# load dataset
adata = scvi.data.heart_cell_atlas_subsampled()

# preprocess dataset
sc.pp.filter_genes(adata, min_counts=3)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=1200,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
)

# specify name of dataset 
data_name = 'HeartAtlas'

# specify attributes
cats = ['cell_type', 'cell_source', 'gender', 'region']

# create numerical index for each attr in cats
create_cats_idx(adata, cats)

today = datetime.today().strftime('%Y-%m-%d')

# create directory with models
isExist = os.path.exists("models")
if not isExist:
    os.makedirs("models")
    print("The models directory is created!")
else:
    print("The models directory already exists!")


# ---------  specify hyperparamters ----------------
# train params
mode=(0,1,2,3,4)

# architecture params
n_layers=1

train_dict = {'max_epochs': epochs, 'batch_size': batch_size, 'cf_weight': cf_weight,
              'beta': beta, 'clf_weight': clf_weight, 'adv_clf_weight': adv_clf_weight,
              'adv_period': adv_period, 'n_cf': n_cf}  #'mode': mode, 

module_name = 'dis2p'
pre_path = f'models/{module_name}'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)

# specify a name for your model
model_name =  f'{today},{module_name},{data_name},' + f'n_layers={n_layers},' + ','.join(k + '=' + str(v) for k, v in train_dict.items())


# -----------  train DiffairVI  ----------------------
# load model (if trained before)
try:
    model = Dis2pVI.load(f"{pre_path}/{model_name}", adata=adata)

# trains the model (if not trained before) and save it into: pre_path + model_name
except:

    Dis2pVI.setup_anndata(
        adata,
        layer='counts',
        categorical_covariate_keys=cats,
        continuous_covariate_keys=[]
    )
    model = Dis2pVI(adata, n_layers=n_layers)
    model.train(**train_dict)
    model.save(f"{pre_path}/{model_name}")


# ------------  retrieve latents -------
# Z_0
adata.obsm[f'{module_name}_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

for i in range(len(cats)):
    null_idx = [s for s in range(len(cats)) if s != i]
    # Z_i
    adata.obsm[f'{module_name}_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
    # Z_{-i}
    adata.obsm[f'{module_name}_Z_not_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=[i], nullify_shared=False)


# -------  Disentanglement/Fainess metrics  --------------
# MIG, MIPG
MI, MI_not_max, MI_not, MI_dif_max, MI_dif, maxMIG, concatMIG = Mixed_KSG_MI_metrics(adata, cats, module_name)

# DP, EO
create_cats_idx(adata, ['NRP'])
y_name = 'NRP_idx'
acc, DP_diff, EO_diff = fair_clf_metrics(adata, cats, y_name, module_name)

# classifier Si
acc, acc_not_concat, acc_not_max, acc_gap_concat, acc_gap_max = clf_S_Z_metrics(adata, cats, module_name)


# convert results to dataframes
df_MI = pd.DataFrame(
      [MI, MI_not_max, MI_not, MI_dif_max, MI_dif]
)

df_fairness = pd.DataFrame(
      [acc, DP_diff, EO_diff]
)

df_acc = pd.DataFrame(
      [acc, acc_not_concat, acc_not_max, acc_gap_concat, acc_gap_max]
)

# create output folder and save csv
metrics_dir = f'metrics/{module_name}/'
os.makedirs(metrics_dir, exist_ok=True)  
df_acc.to_csv(metrics_dir+model_name+',acc.csv')
df_fairness.to_csv(metrics_dir+model_name+',fair.csv') 
df_MI.to_csv(metrics_dir+model_name+',MI.csv')


# -------  OOD predictions  ----------------------------------

# OOD params
cov_idx = 2
other_covs_values = ('Ventricular_Cardiomyocyte', 'Sanger-Nuclei', 'RV')
cov_value = 'Male'
cov_value_cf = 'Female'

new_model_name = 'OOD,' + f'{today},{module_name},{data_name},' + f'n_layers={n_layers},' + ','.join(k + '=' + str(v) for k, v in train_dict.items())

true_x_counts_mean, true_x_counts_variance, px_cf_mean_pred, px_cf_variance_pred = ood_for_given_covs(
        adata=adata,
        cats=cats,
        vi_cls=Dis2pVI,
        model_name=new_model_name,
        pre_path=pre_path,
        cov_idx=cov_idx,
        cov_value=cov_value,
        cov_value_cf=cov_value_cf,
        other_covs_values=other_covs_values,
        remove_all_samples_with_other_covs_values=True,
        **train_dict,
)

true_x_counts_mean_np = true_x_counts_mean.detach().numpy()
true_x_counts_variance_np = true_x_counts_variance.detach().numpy()
px_cf_mean_pred_np = px_cf_mean_pred.detach().numpy()
px_cf_variance_pred_np = px_cf_variance_pred.detach().numpy()

# convert results to dataframes
df_ood = pd.DataFrame(
      [true_x_counts_mean_np, true_x_counts_variance_np, px_cf_mean_pred_np, px_cf_variance_pred_np]
)

# create output folder and save csv
metrics_dir = f'metrics/{module_name}'
os.makedirs(metrics_dir, exist_ok=True)  
df_ood.to_csv(metrics_dir+'/'+new_model_name+',ood_pred.csv')


# -------  OOD R2 metrics  ----------------------------------

n_top_deg = 20
cov_name = cats[cov_idx]

print()
print('R2 means metrics')
mean_r2, mean_r2_log, mean_r2_deg, mean_r2_log_deg = r2_eval(adata, cov_name, cov_value_cf, true_x_counts_mean, px_cf_mean_pred, n_top_deg=n_top_deg)

print()
print('R2 variance metrics')
var_r2, var_r2_log, var_r2_deg, var_r2_log_deg = r2_eval(adata, cov_name, cov_value_cf, true_x_counts_variance, px_cf_variance_pred, n_top_deg=n_top_deg)


# convert results to dataframes
df_r2 = pd.DataFrame(
      [[mean_r2, mean_r2_log, mean_r2_deg, mean_r2_log_deg], [var_r2, var_r2_log, var_r2_deg, var_r2_log_deg]]
)

# create output folder and save csv
metrics_dir = f'metrics/{module_name}'
os.makedirs(metrics_dir, exist_ok=True)
df_r2.to_csv(metrics_dir+'/'+model_name+',r2.csv')

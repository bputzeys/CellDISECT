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
import os
import argparse
import sys


# append paths to sys.path to import from many locations (eg evaluate and scib)
current_script_path = os.path.dirname(__file__)
fpath = os.path.join(current_script_path, "..")
sys.path.append(fpath)
from evaluation.metrics import *
from evaluation.evaluate import *
from scib_metrics_dev.src.scib_metrics.benchmark import Benchmarker



# ------------  READ INPUTS  -------------------
parser = argparse.ArgumentParser(
                    prog='scFAIR',
                    description='train scFAIR specifying data, hyperparameters, destinations',
                    epilog='Text at the bottom of help')

parser.add_argument('-e', '--epochs',  type=int, default=400,
                    help='an integer for the number of epochs')
parser.add_argument('-b', '--batch', type=int, default=256,
                    help='an integer for the batch size')
parser.add_argument('-cfw', '--CFwei', type=float, default=2.0,
                    help='a float for the counterfactual weight')
parser.add_argument('-a', '--alpha', type=float, default=2.0,
                    help='a float for the number of epochs')
parser.add_argument('-CLFw', '--classwei', type=float, default=50.0,
                    help='a float for the weight of classification loss term')
parser.add_argument('-advCLFw', '--advclasswei', type=float, default=10.0,
                    help='a float for the weight of adversarial classification loss term')
parser.add_argument('-advPER', '--advperiod', type=int, default=1,
                    help='an integer for the adversarial period')
parser.add_argument('-m', '--mode', type=int, nargs="+",
                    help='a integers for the loss terms to have')
parser.add_argument('-ngenes', '--ngenes', type=int, default=1000,
                    help='a integers for the loss terms to have')


args = parser.parse_args()

print(args)
epochs = args.epochs
batch_size = args.batch
cf_weight = args.CFwei
alpha = args.alpha
cl_weight = args.classwei
adv_cl_weight = args.advclasswei
adv_period = args.advperiod
mode=args.mode
mode=tuple(args.mode)

print("epochs are")
print(epochs)
print("mode is")
print(mode)


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
data_name = 'heartAtlas'

# specify attributes
cats = ['cell_type', 'cell_source', 'gender', 'region']

# specify a path that will be used to save any trained model later (directories in the path should be created first)
pre_path = f"models/FairVI"

# specify a path that will be used to save the preprocessed indices of paired samples for the counterfactual term calculation
idx_cf_tensor_path = f'idx_cf_tensors/heart_4cov'

# create numerical index for each attr in cats
create_cats_idx(adata, cats)

today = datetime.today().strftime('%Y-%m-%d')


# create folders for scfair outputs
isExist = os.path.exists(idx_cf_tensor_path)
if not isExist:
    os.makedirs(idx_cf_tensor_path)
    print("The CF directory is created!")
else:
    print("The CF directory already exists!")

# create directory with models
isExist = os.path.exists("models")
if not isExist:
    os.makedirs("models")
    print("The models directory is created!")
else:
    print("The models directory already exists!")


    
# ---------  specify hyperparamters ----------------
epochs = 400
batch_size = 256
cf_weight = 2
alpha = 1
clf_weight = 50
adv_clf_weight = 10
adv_period = 1
mode=(0,1,2,3,4)

train_dict = {'max_epochs': epochs, 'batch_size': batch_size, 'cf_weight': cf_weight,
              'alpha': alpha, 'clf_weight': clf_weight, 'adv_clf_weight': adv_clf_weight,
              'adv_period': adv_period, 'mode': mode}

# specify a name for your model
model_name = today + ',' + data_name + ',' + ','.join(cats) + ',' + ','.join(k + '=' + str(v) for k, v in train_dict.items())



# -----------  train FairVI  ----------------------
# load model (if trained before)
try:
    model = FairVI.load(f"{pre_path}/{model_name}", adata=adata)
# trains the model (if not trained before) and save it into: pre_path + model_name
except:
    FairVI.setup_anndata(
        adata,
        layer='counts',
        categorical_covariate_keys=cats,
        continuous_covariate_keys=[]
    )
    model = FairVI(adata, idx_cf_tensor_path=idx_cf_tensor_path)
    model.train(**train_dict)
    model.save(f"{pre_path}/{model_name}")

model.idx_cf_tensor_path = idx_cf_tensor_path


# ------------  retrieve latents and plot and save UMAPS -------
# trains the model (if not trained before) and save it into: pre_path + model_name
# then get latent representaion (they will be strored in adata.obsm)
# Z_0 = adata.obsm["Z_0"]
# Z_i =  adata.obsm["Z_i"] for i in [1, ..., len(cats)]
# Z_{-i} = adata.obsm["Z _not_i"] for i in [1, ..., len(cats)]
model, adata = latent(
        adata = adata,
        cats = cats,
        new_model_name = model_name,
        pre_path = pre_path,
        idx_cf_tensor_path = idx_cf_tensor_path,
        plot_umap = True,
        **train_dict,
)



# -------  Disentanglement/Fainess metrics  --------------
# classifier Si | Zi
acc_results_1 = clf_S_Z_metrics(adata, cats)
# classifier Si | (Z - Zi)
acc_results_2 = clf_S_Z_not_metrics(adata, cats)

# fairness metrics: DP, EO, ...
create_cats_idx(adata, ['NRP'])
y_name = 'NRP_idx'
ACC, DP_diff, EO_diff = fair_clf_metrics(adata, cats, y_name)

# Max Mutual Information by taking Max over Dims
MI_md, MI_not_md, mig_md = max_dim_MI_metrics(adata, cats)
# Mutual Information by Mixed_KSG (https://github.com/wgao9/mixed_KSG/blob/master/mixed.py)
#MI, MI_not, MI_not_max, mig, mipg = Mixed_KSG_MI_metrics(adata, cats)

# convert results to dataframes
df_acc_results_1 = pd.DataFrame(
      acc_results_1,
      index=[f"S{i+1}" for i in range(len(acc_results_1))],
      columns=['train_acc', 'test_acc'])

df_acc_results_2 = pd.DataFrame(
      acc_results_2,
      index=[f"S{i+1}" for i in range(len(acc_results_2))],
      columns=['train_acc', 'test_acc'])

df_fairness = pd.DataFrame(
      { "ACC":     ACC,
        "DP_diff": DP_diff,
        "EO_diff": EO_diff},
      index=[f"{i+1}" for i in range(len(ACC))]
)


# create output folder and save csv
os.makedirs('data/output', exist_ok=True)  
df_acc_results_1.to_csv('data/output/'+model_name+'acc_cl.csv', index=True)  
df_acc_results_2.to_csv('data/output/'+model_name+'acc_adv_cl.csv', index=True)  
df_fairness.to_csv('data/output/'+model_name+'fairness_metrics.csv', index=True)  


# # -------  OOD metrics  ----------------------------------
# cov_idx = 2  # index of target attribute in cats
# cov_value = 'Male'  # factual value for the target attribute
# cov_value_cf = 'Female'  # counterfactual value for the target attribute
# other_covs_values = ('Ventricular_Cardiomyocyte', 'Sanger-Nuclei', 'RV')  # fixed values for other attibutes that we perform OOD on them
# n_top_deg = 100  # number of top DE genes for R2 

# # holds-out all cells with other_covs_values (prefered method for OOD)
# # splits other cells to train/validation sets, trains the model, and finally perform OOD on held-out cells
# # returns: 
# #     1) true genes counts vector by averaging all held-out cells
# #     2) predicted genes counts vector for average of held-out cells
# true_x_counts_mean, px_cf_mean_pred = ood_for_given_covs_2(
#         adata=adata,
#         cats=cats,
#         new_model_name=model_name,
#         pre_path=pre_path,
#         idx_cf_tensor_path=idx_cf_tensor_path,
#         cov_idx=cov_idx,
#         cov_value=cov_value,
#         cov_value_cf=cov_value_cf,
#         other_covs_values=other_covs_values,
#         n_top_deg=n_top_deg,
#         **train_dict,
# )









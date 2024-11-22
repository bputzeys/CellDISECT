<img src="https://github.com/Lotfollahi-lab/celldisect/blob/main/media/CellDISECT_Logo_whitebg.png" width="1000" alt="celldisect-logo">

[![PyPI version](https://badge.fury.io/py/celldisect.svg)](https://badge.fury.io/py/celldisect)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Lotfollahi-lab/celldisect/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/Lotfollahi-lab/celldisect?logo=GitHub&color=yellow)](https://github.com/Lotfollahi-lab/celldisect/stargazers)

[comment]: [![PyPIDownloads](https://static.pepy.tech/badge/celldisect)](https://pepy.tech/project/celldisect)

[comment]: [![Docs](https://readthedocs.org/projects/celldisect/badge/?version=latest)](https://celldisect.readthedocs.io/en/stable/?badge=stable)

# Cell DISentangled Experts for Covariate counTerfactuals (CellDISECT)
CellDISECT is a causal generative model designed to disentangle known covariate variations from unknown ones at test time while simultaneously learning to make counterfactual predictions. CellDISECT finds multiple latent representations for each cell, one unsupervised, and the rest weakly supervised by the provided covariates. 

Its latent space captures not only covariate-specific information but also new biology, thereby offering users a multifaceted view of the data and enhancing the ability for cell type discovery. 

Moreover, by using different "expert" models to learn each latent it achieves flexible fairness in single-cell analysis. This flexibility allows choosing which covariates to use as biological and which as batch, at test time, as opposed to at train time like with most methods.

Finally, it can model the effect of perturbations on one or many covariates by calculating the counterfactual gene expression under the perturbations.


<p align="center">
  <img src="https://github.com/Lotfollahi-lab/celldisect/blob/main/media/celldisect_illustration.png" width="750">
</p>


Installation
============

Prerequisites
--
Conda Environment
--
We recommend using [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a conda environment for using CellDISECT. You can create a python environment using the following command:

    conda create -n CellDISECT python=3.9

Then, you can activate the environment using:

    conda activate CellDISECT


- Install pytorch (This version of CellDISECT is tested with pytorch 2.1.2 and cuda 12, install the appropriate version of pytorch for your system.)
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

- (Optional) if you plan to use RAPIDS/rapids-singlecell:
```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.4.* dask-cudf-cu12==24.4.* cuml-cu12==24.4.* \
    cugraph-cu12==24.4.* cuspatial-cu12==24.4.* cuproj-cu12==24.4.* \
    cuxfilter-cu12==24.4.* cucim-cu12==24.4.* pylibraft-cu12==24.4.* \
    raft-dask-cu12==24.4.* cuvs-cu12==24.4.*

pip install rapids-singlecell
```

- Install CellDISECT
You can either install the stable version published on pypi using pip:
```
pip install celldisect
```
Or you can install the latest developed version directly from our github:
```
pip install git+https://github.com/Lotfollahi-lab/CellDISECT
```

- (Optional) to install cuda enabled jax:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


How to use CellDISEC
===

|Description | Link |
| --- | --- |
| Counterfactual prediction and DEG analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lotfollahi-Lab/CellDISECT/blob/main/tutorials/CellDISECT_Counterfactual.ipynb) - [![Open In Github](https://img.shields.io/badge/docs-blue)](https://github.com/Lotfollahi-Lab/CellDISECT/blob/main/tutorials/CellDISECT_Counterfactual.ipynb) |

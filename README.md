# Disentangling covariates to predict counterfacctuals (dis2p)
Causal generative model designed to disentangle known covariate variations from unknown ones at test time while simultaneously learning to make counterfactual predictions.


Installation
============

Prerequisites
--
Conda Environment
--
We recommend using [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a conda environment for using Dis2P. You can create a python environment using the following command:

    conda create -n dis2p python=3.9

Then, you can activate the environment using:

    conda activate dis2p


Install pytorch (This version of dis2p is tested with pytorch 2.1.0 and cuda 11.8, install the appropriate version of pytorch for your system.)
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Install the latest version of dis2p
```
pip install git+https://github.com/Lotfollahi-lab/dis2p
```

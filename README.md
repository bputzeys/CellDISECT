# Cell DISentangled Experts for Covariate counTerfactuals (CellDISECT)
Causal generative model designed to disentangle known covariate variations from unknown ones at test time while simultaneously learning to make counterfactual predictions.


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

- Install the latest version of CellDISECT
```
pip install git+https://github.com/Lotfollahi-lab/CellDISECT
```

- (Optional) to install cuda enabled jax:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


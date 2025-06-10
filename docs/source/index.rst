=======================================
Welcome to CellDISECT's documentation!
=======================================

.. image:: _static/images/CellDISECT_Logo_whitebg.png
   :align: center
   :width: 800px

.. raw:: html

   <div style="text-align: center; margin: 20px 0;">
     <a href="https://pypi.org/project/celldisect" target="_blank">
       <img src="https://img.shields.io/pypi/v/celldisect?label=pypi%20package&color=green" alt="PyPI Version">
     </a>
     <a href="https://github.com/Lotfollahi-Lab/CellDISECT/blob/main/LICENSE" target="_blank">
       <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue" alt="License">
     </a>
     <a href="https://github.com/Lotfollahi-Lab/CellDISECT/stargazers" target="_blank">
       <img src="https://img.shields.io/github/stars/Lotfollahi-Lab/CellDISECT?logo=github" alt="GitHub Stars">
     </a>
     <a href="https://pepy.tech/projects/celldisect">
       <img src="https://static.pepy.tech/badge/celldisect" alt="PyPI Downloads">
      </a>
     <a href="https://www.biorxiv.org/content/10.1101/2025.06.03.657578v1" target="_blank">
       <img src="https://img.shields.io/badge/bioRxiv-2025.06.03.657578-red.svg" alt="bioRxiv Preprint">
     </a>
   </div>

   <div style="text-align: center; margin: 20px 0;">
     <a href="https://github.com/Lotfollahi-Lab/CellDISECT" target="_blank">
       <img src="https://img.shields.io/badge/GitHub-CellDISECT-blue?style=for-the-badge&logo=github" alt="GitHub Repository">
     </a>
   </div>

.. note::
   **Beta Version Available**: A beta version (0.2.0b1) with compatibility for Google Colab and newer versions of torch and scvi-tools is available on the `beta-colab branch <https://github.com/Lotfollahi-Lab/CellDISECT/tree/beta-colab>`_. Install it with ``pip install celldisect==0.2.0b1``.

**CellDISECT** (Cell DISentangled Experts for Covariate counTerfactuals) is a causal generative model designed to disentangle known covariate variations from unknown ones at test time while simultaneously learning to make counterfactual predictions.

.. image:: _static/images/celldisect_illustration.png
   :align: center
   :width: 700px

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   examples
   tutorials/index
   api/index
   changelog
   contributing
   references

Installation
-------------

Prerequisites
~~~~~~~~~~~~~~

We recommend using `Anaconda <https://www.anaconda.com/>`_/`Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_ to create a conda environment for using CellDISECT.

1. Create and activate a conda environment:

.. code-block:: bash

   conda create -n CellDISECT python=3.9
   conda activate CellDISECT

2. Install PyTorch (tested with pytorch 2.1.2 and cuda 12):

.. code-block:: bash

   conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

3. Install CellDISECT:

You can install the stable version using pip:

.. code-block:: bash

   pip install celldisect

Or install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/Lotfollahi-lab/CellDISECT

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For RAPIDS/rapids-singlecell support:

.. code-block:: bash

   pip install \
       --extra-index-url=https://pypi.nvidia.com \
       cudf-cu12==24.4.* dask-cudf-cu12==24.4.* cuml-cu12==24.4.* \
       cugraph-cu12==24.4.* cuspatial-cu12==24.4.* cuproj-cu12==24.4.* \
       cuxfilter-cu12==24.4.* cucim-cu12==24.4.* pylibraft-cu12==24.4.* \
       raft-dask-cu12==24.4.* cuvs-cu12==24.4.*
   
   pip install rapids-singlecell

For CUDA-enabled JAX:

.. code-block:: bash

   pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Quick Start
------------

Here's a simple example to get you started:

.. code-block:: python

   from celldisect import CellDISECT
   import scanpy as sc

   # Load your data
   adata = sc.read_h5ad('your_data.h5ad')
   adata.X = adata.layers['counts'].copy()
   cats = ['cov1', 'cov2']
   cell_type_included = False
   # Initialize and train the model
   CellDISECT.setup_anndata(
    adata,
    layer='counts',
    categorical_covariate_keys=cats,
    continuous_covariate_keys=[],
    add_cluster_covariate=not cell_type_included, # add_cluster_covariate if cell type is not included
   )
   model = CellDISECT(adata)
   model.train()

   # Make predictions
   predictions = model.predict_counterfactuals(
       adata,
       cov_names=['cov1'],
       cov_values=['val1'],
       cov_values_cf=['val2'],
       cats=cats,
   )

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search` 
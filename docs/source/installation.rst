Installation
============

We recommend using `Anaconda <https://www.anaconda.com/>`_/`Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_ to create a conda environment for using CellDISECT.

1. Create and activate a conda environment:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n CellDISECT python=3.9
   conda activate CellDISECT

2. Install PyTorch (tested with pytorch 2.1.2 and cuda 12):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

3. Install CellDISECT:
~~~~~~~~~~~~~~~~~~~~~~
You can install the stable version using pip:

.. code-block:: bash

   pip install celldisect

Install the beta version with Google Colab and newer dependency support:

.. code-block:: bash

   pip install celldisect==0.2.0b1

Or install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/Lotfollahi-lab/CellDISECT

Or install from local directory by cloning the repository for development:

.. code-block:: bash

   git clone https://github.com/Lotfollahi-lab/CellDISECT.git
   cd CellDISECT
   pip install -e .

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


Dependencies
------------

CellDISECT has the following main dependencies:

* anndata (>=0.10.8, <0.10.9)
* scvi-tools (>=0.20.3, <1.0.0)
* torch (>=2.1.0, <2.3.0)
* scanpy
* numpy (>=1.26.3, <1.27.0)
* jax (>=0.4.16, <0.4.24)
* lightning (>=2.2.0, <2.3.0)

For a complete list of dependencies, please refer to the pyproject.toml file in the repository. 
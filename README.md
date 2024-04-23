# Disentangling covariates to predict counterfacctuals (dis2p)
Causal generative model designed to disentangle known covariate variations from unknown ones at test time while simultaneously learning to make counterfactual predictions.


# Installation
If you have installed an older version of dis2p, uninstall it
```
pip uninstall dis2p
```

Install pytorch (This version of dis2p is tested with pytorch 2.0.0 and cuda 11.7, install the appropriate version of pytorch for your system.)
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install the latest version of dis2p
```
pip install git+https://github.com/Lotfollahi-lab/dis2p
```


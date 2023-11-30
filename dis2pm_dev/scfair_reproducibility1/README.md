# scfair-reproducibility
results of scfair for reproducibility

# create environment for scfair-reproducibility
```
conda create --prefix /your/path/scfair_analysis_env python=3.10 -y
conda activate /your/path/scfair_analysis_env
pip install torch torchvision torchaudio
pip install scvi-tools
pip install ipykernel
python -m ipykernel install --user --name scfair_analysis_env
pip install matplotlib
pip install scanpy
pip install xgboost
pip install fairlearn
pip install git+https://github.com/M0hammadL/scfair.git
pip install scib-metrics
pip install scikit-learn
pip install scikit-misc
```
# running the analysis
```
clone this repository and then from the root directory run the following commands:
```
## create the hyperparameter options to be sweeped
```
python scripts/hyperparameter_grid.py
```
## read off the number of lines in the output files as
```
wc -l ta/output/hyperparameter_grid/hyperparameter_grid.csv
```
## submit jobs to the farm one for each line in the hyperparam file
```
#bsub -J "arrayName[1-999]" -G teichlab  < scripts/job_eDm_heart_adult.sh
#bsub -J "arrayName[1-(number_of_lines-1) ]" -G teichlab -q gpu-normal -n 8 -M120000 -R"select[mem>120000] rusage[mem=120000] span[hosts=1]" -gpu"mode=shared:j_exclusive=no:gmem=10000:num=1"  < scripts/bsub_fairvi.sh
bash scripts/bash_for_many_jobs.sh  
```

# clone
```
git clone https://github.com/M0hammadL/scfair-reproducibility.git
```

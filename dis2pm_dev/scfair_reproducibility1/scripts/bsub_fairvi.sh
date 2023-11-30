#!/bin/bash

# To run:
# the metadata file needs to be i) tsv, 
#                               ii) in the present directory, and 
#                               iii) have the sample paths in the last column
# wc -l metadata_heart.csv
# then use number_of_lines-1 as array number
# bsub -J "arrayName[1-(number_of_lines-1) ]" -G teichlab  < job_eDm_heart_adult.sh

# LSF configuration
#BSUB -J fairvi       
#BSUB -n 1                  # number of cores
#BSUB -q normal             # queue
#BSUB -R "select[mem>50000] rusage[mem=50000]"
#BSUB -M 50000
#BSUB -o logs/output_%J_%I.out #output. %J is job-id %I is job-array index
#BSUB -e logs/error_%J_%I.err

mkdir -p logs  # for logs per job

CFweis=($(awk -F$',' '{if (NR!=1) {print $1}}'  data/output/hyperparameter_grid/hyperparameter_grid.csv))
classweis=($(awk -F$',' '{if (NR!=1) {print $2}}'  data/output/hyperparameter_grid/hyperparameter_grid.csv))
advclassweis=($(awk -F$',' '{if (NR!=1) {print $2}}'  data/output/hyperparameter_grid/hyperparameter_grid.csv))
echo $LSB_JOBINDEX
echo ${CFweis[LSB_JOBINDEX-1]}
echo ${classweis[LSB_JOBINDEX-1]}
echo ${advclassweis[LSB_JOBINDEX-1]}

python train_and_metrics.py -m 4 --CFwei ${CFweis[LSB_JOBINDEX-1]} --classwei ${classweis[LSB_JOBINDEX-1]} --advclasswei ${advclassweis[LSB_JOBINDEX-1]} 



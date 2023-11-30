export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p logs  # for logs per job

CFweis=($(awk -F$',' '{if (NR!=1) {print $2}}'  data/output/hyperparam_grid/hyperparam_grid.csv))
classweis=($(awk -F$',' '{if (NR!=1) {print $3}}'  data/output/hyperparam_grid/hyperparam_grid.csv))
advclassweis=($(awk -F$',' '{if (NR!=1) {print $4}}'  data/output/hyperparam_grid/hyperparam_grid.csv))

# count number of lines in csv
tmp=6 # $(wc -l <data/output/hyperparam_grid/hyperparam_grid.csv)-1  # if you want to just submit the first 32 jobs change this line to 32

# for every line submit a job
for ((i=0; i<$tmp; i++)) 
do
    echo $i
    echo ${CFweis[i]}
    echo ${classweis[i]}
    echo ${advclassweis[i]}
    #try gentler shared GPU syntax
    bsub -J cbk -G teichlab -o 'logs/cbk.%J.out' -e 'logs/cbk.%J.err' -q gpu-normal -n 1 -M 15000 -R"select[mem>15000] rusage[mem=15000] span[hosts=1]" -gpu"mode=shared:j_exclusive=no:gmem=10000:num=1" "python scripts/train_heart.py --CFwei ${CFweis[LSB_JOBINDEX-1]} --classwei ${classweis[LSB_JOBINDEX-1]} --advclasswei ${advclassweis[LSB_JOBINDEX-1]} -e 400 -b 128 -be 1 -advPER 1 -n_cf 1"
done








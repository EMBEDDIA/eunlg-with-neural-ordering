#!/bin/bash
#SBATCH -M ukko2
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH,TORCH_HOME,BULK,LOCS,PL
#SBATCH -J gen
#SBATCH --chdir=/wrk/users/eliel/projects/embeddia/eunlg
#SBATCH -o /wrk/users/eliel/projects/embeddia/eunlg/jobs/res/%A_%a.txt
#SBATCH -e /wrk/users/eliel/projects/embeddia/eunlg/jobs/err/%A_%a.txt
#SBATCH -t 10:00:00
#SBATCH -c 5
#SBATCH --mem=10G
#SBATCH --mail-type=END

module purge
module load Python/3.7.0-intel-2018b
module load CUDA/10.1.105

V=(neural_filter neural_filter_ctx_setpen neural_filter_ctx neural_filter_setpen baseline_filter list_baseline \
list_neural)
LOC=(DE FI EE AT HR SE)
ID=SLURM_ARRAY_TASK_ID

# parameters (model, sequence length for scoring, coefficient scale given in $BULK)

echo "BULK, PL, LOCS are: "
echo $BULK
echo $PL
echo $LOCS
srun $USERAPPL/ve37/bin/python3 eunlg/bulk_generate.py -l en -o $BULK -v $PL -d cphi --locations ${LOC[$ID]} \
--verbose


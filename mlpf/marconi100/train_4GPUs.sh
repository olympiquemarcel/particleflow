#!/bin/sh

# Walltime limit
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p m100_usr_prod
#SBATCH --gpus 4
#SBATCH --account=Ppp4x_5710
#SBATCH --exclusive

# Job name
#SBATCH -J train

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err
#!/bin/sh

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module purge
module load profile/deeplrn autoload cineca-ai/2.0.0
nvidia-smi

chprj -d Ppp4x_5710

mkdir $TMPDIR/particleflow
rsync -ar --exclude={".git","experiments"} . $TMPDIR/particleflow
cd $TMPDIR/particleflow
if [ $? -eq 0 ]
then
  echo "Successfully changed directory"
else
  echo "Could not change directory" >&2
  exit 1
fi
mkdir experiments

echo 'Starting training.'
# Run the training of the base GNN model using e.g. 4 GPUs in a data-parallel mode
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c $1 -p $2
echo 'Training done.'

rsync -a experiments/ /m100_work/Ppp4x_5710/experiments/

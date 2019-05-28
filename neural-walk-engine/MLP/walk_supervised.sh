#PBS -l walltime=06:00:00
#PBS -o ppo_${PBS_JOBID}-o.txt
#PBS -e ppo_${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR

source activate learning-3d

N_LAYERS=$1
LAYER_SIZE=$2
ACTIVATION=$3
LEARNING_RATE=$4
LOSS_FUNCTION=$5
BATCH_SIZE=$6
EPOCHS_DROP=$7
DROP=$8


python walk_supervised.py --n_layers $N_LAYERS --layer_size $LAYER_SIZE --activation $ACTIVATION --learning_rate $LEARNING_RATE --loss_function $LOSS_FUNCTION --batch_size $BATCH_SIZE --epochs_drop $EPOCHS_DROP --drop $DROP

echo "Finishing Job"
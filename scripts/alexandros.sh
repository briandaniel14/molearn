#!/bin/bash

#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 2:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4

set -e # fail fast

export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export MOLEARN_PATH=/home/${USER}/repos/molearn
export SCRATCH_HOME=/disk/scratch/${USER}
export DATA_HOME=${PWD}/../data
export DATA_SCRATCH=${SCRATCH_HOME}/experiments/data_${TIMESTAMP}
export OUTPUT_DIR=${SCRATCH_HOME}/experiments/experiment_${TIMESTAMP}


# Check for the -g flag
if [[ "$*" == *"-g"* ]]; then
    CONDA_ENV_NAME="molearn-gpu"
else
    CONDA_ENV_NAME="molearn"
fi

source /home/${USER}/miniconda3/bin/activate molearn

echo "Activated ${CONDA_ENV_NAME}"

if python3 -c "import molearn" 2>/dev/null; then
    echo "Molearn is already installed."
else
    echo "Molearn not found. Installing from source..."
    python3 -m pip install "$MOLEARN_PATH"
    echo "Molearn nstallation complete."
fi

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}."
echo "I'm running on ${SLURM_JOB_NODELIST}."
echo "Job started at ${dt}."

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
mkdir -p ${DATA_SCRATCH}
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}

mkdir -p ${OUTPUT_DIR}
echo "Created ${OUTPUT_DIR}"

echo "Running Python script..."
python3 src/drifts_experiment.py \
    --checkpoint_file=${DATA_SCRATCH}/models/checkpoints/foldingnet/checkpoint_no_optimizer_state_dict_epoch167_loss0.003259085263643.ckpt \
    --data_path=${DATA_SCRATCH}/proteins \
    --num_iters=100 \
    --grid_scale_factor=0.5 \
    --resolution=50 \
    --pdbs MurD_closed.pdb MurD_open.pdb \
    --output_dir=${OUTPUT_DIR} \
    --autoencoder=fold_net

OUTPUT_HOME=${PWD}/
mkdir -p ${OUTPUT_HOME}
rsync 
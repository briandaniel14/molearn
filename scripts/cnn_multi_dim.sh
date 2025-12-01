#!/bin/bash

export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export MOLEARN_PATH=/home/${USER}/repos/molearn
export SCRATCH_HOME=/disk/scratch/${USER}/molearn
export OUTPUT_SCRATCH=${SCRATCH_HOME}/examples/cnn_multi_dim_checkpoints
export OUTPUT_DIST=${MOLEARN_PATH}/results/${TIMESTAMP}

mkdir -p $OUTPUT_DIST

mkdir -p ${SCRATCH_HOME}
rsync --archive --update --compress --progress ${MOLEARN_PATH}/ ${SCRATCH_HOME}

echo 'unzipping data'  
cd ${SCRATCH_HOME}/examples/data

tar -xzf MurD_closed_apo.tar.gz
tar -xzf MurD_closed.tar.gz
tar -xzf MurD_open.tar.gz

cd ..

source /home/${USER}/miniconda3/bin/activate molearn

echo 'running script'
python runner_CNN_multi_dim.py

rsync --archive --update --compress --progress ${OUTPUT_SCRATCH} ${OUTPUT_DIST}

echo 'cleaning up'
cd /disk/scratch/
rm -r -f s2305437
echo 'cleanup complete'

echo "Job is done"
exit 0
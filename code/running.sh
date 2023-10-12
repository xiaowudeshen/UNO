#!/bin/bash
#PBS -N T5-ntest
#PBS -l select=1:ncpus=16:ngpus=1:mem=100gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -P 11003552
#PBS -q normal
#PBS -o log/JT_test.log

# The following environment variables will be preset after your job is submitted and started.
################################################# 
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
#################################################


source activate T5


cd $PBS_O_WORKDIR
# cd alpaca_finetuning_v1
[ -d log ] || mkdir log  

nvidia-smi

python T5.py --train_batch_size 8 --GPU 1 --n_epochs 1 --model_checkpoint t5-small --saving_dir t5_small --slot_lang question --except_domain hotel --joint_training mask_slot 

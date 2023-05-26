#!/bin/sh
### Set the job name (for your reference)
#PBS -N COL775_A1
### Set the project name, your department code by default
#PBS -P textile
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M tt1170896@iitd.ac.in
####
#PBS -l select=1:ncpus=1:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=7:00:00
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
cd ~/
module load apps/anaconda/3
source activate icswm
module unload apps/anaconda/3
cd scratch/COL775/A1/src
python main.py --task train --args experiment_name adagrad_const_gn optimizer Adagrad  root ./../experiment/part_1_2  normalization gn
python main.py --task train --args experiment_name adagrad_const_in optimizer Adagrad  root ./../experiment/part_1_2  normalization in
python main.py --task train --args experiment_name adagrad_const_bin optimizer Adagrad  root ./../experiment/part_1_2  normalization bin
python main.py --task train --args experiment_name adagrad_const_ln optimizer Adagrad  root ./../experiment/part_1_2  normalization ln
python main.py --task train --args experiment_name adagrad_const_inbuilt optimizer Adagrad  root ./../experiment/part_1_2  normalization inbuilt
python main.py --task train --args experiment_name adagrad_const_none optimizer Adagrad  root ./../experiment/part_1_2  normalization none
python main.py --task train --args experiment_name adagrad_const_bn optimizer Adagrad  root ./../experiment/part_1_2  normalization bn


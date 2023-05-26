#!/bin/sh
### Set the job name (for your reference)
#PBS -N COL775_A1_exponential
### Set the project name, your department code by default
#PBS -P textile
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M tt1170896@iitd.ac.in
####
#PBS -l select=1:ncpus=1:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=6:00:00
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
cd ~/
module load apps/anaconda/3
source activate icswm
module unload apps/anaconda/3
cd scratch/COL775/A1/src
python main.py --task train --args   root ./../experiments/part_1_1   experiment_name sgd_linear lr_scheduler.name linear 
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name sgd_mom_linear lr_scheduler.name linear momentum 0.9 
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name adam_linear optimizer Adam lr_scheduler.name linear 
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name rmsprop_linear optimizer RMSprop lr_scheduler.name linear 
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name adagrad_linear optimizer Adagrad lr_scheduler.name linear 

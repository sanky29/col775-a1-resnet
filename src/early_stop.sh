python main.py --task train --args   root ./../experiments/part_1_1  experiment_name sgd_exponential lr_scheduler.name exponential
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name sgd_mom_exponential lr_scheduler.name exponential momentum 0.9
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name adam_exponential optimizer Adam lr_scheduler.name exponential
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name rmsprop_exponential optimizer RMSprop lr_scheduler.name exponential
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name adagrad_exponential optimizer Adagrad lr_scheduler.name exponential 

python main.py --task train --args   root ./../experiments/part_1_1  experiment_name sgd_step lr_scheduler.name step
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name sgd_mom_step lr_scheduler.name step momentum 0.9
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name adam_step optimizer Adam lr_scheduler.name step
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name rmsprop_step optimizer RMSprop lr_scheduler.name step
python main.py --task train --args   root ./../experiments/part_1_1   experiment_name adagrad_step optimizer Adagrad lr_scheduler.name step

python main.py --task train --args   root ./../experiments/part_1_1   experiment_name sgd_linear lr_scheduler.name linear
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name sgd_mom_linear lr_scheduler.name linear momentum 0.9
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name adam_linear optimizer Adam lr_scheduler.name linear
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name rmsprop_linear optimizer RMSprop lr_scheduler.name linear
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name adagrad_linear optimizer Adagrad lr_scheduler.name linear

python main.py --task train --args   root ./../experiments/part_1_1 experiment_name sgd_constant
python main.py --task train --args   root ./../experiments/part_1_1 experiment_name sgd_mom_constant momentum 0.9
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name adam_constant optimizer Adam 
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name rmsprop_constant optimizer RMSprop 
python main.py --task train --args   root ./../experiments/part_1_1  experiment_name adagrad_constant optimizer Adagrad 
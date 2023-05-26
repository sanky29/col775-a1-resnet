import subprocess
from config import get_config
from test import Tester
# for v in ['linear','constant','exponential','step']:
#     for o in ['adam','adagrad','sgd','rmsprop', 'sgd_mom']:
#         cmd_str = f'python plot.py --title loss vs epoch for {o} {v} --files ./../experiments/early_stop/{o}_{v}/log.txt ./../experiments/early_stop/{o}_{v}/log.txt --legends train val --column 0 1 --file_name plots/{o}_{v}_early_stop.png'
#         subprocess.run(cmd_str, shell=True)
        

# for v in ['linear','constant','exponential','step']:
#     for o in ['adam','adagrad','sgd','rmsprop', 'sgd_mom']:
#         cmd_str = f'python plot.py --title train loss vs epoch for {o} {v} --files ./../experiments/part_1_1/{o}_{v}/log.txt   --column 0 --file_name plots/{o}_{v}.png'
#         subprocess.run(cmd_str, shell=True)
        
# for o in ['adam','adagrad','sgd','rmsprop', 'sgd_mom']:
#     cmd_str = f'python plot.py --title train loss vs epoch for {o} --files ./../experiments/part_1_1/{o}_constant/log.txt ./../experiments/part_1_1/{o}_linear/log.txt ./../experiments/part_1_1/{o}_exponential/log.txt ./../experiments/part_1_1/{o}_step/log.txt --legends constant linear exponential step --column 0 --file_name plots/{o}.png'
#     subprocess.run(cmd_str, shell=True)

# for o in ['adam','adagrad','sgd','rmsprop', 'sgd_mom']:
#     cmd_str = f'python plot.py --title train loss vs epoch for {o} --files ./../experiments/early_stop/lr_01/{o}_constant/log.txt ./../experiments/early_stop/lr_01/{o}_linear/log.txt ./../experiments/early_stop/lr_01/{o}_exponential/log.txt ./../experiments/early_stop/lr_01/{o}_step/log.txt --legends constant linear exponential step --column 0 --file_name plots/{o}_lr_01.png'
#     subprocess.run(cmd_str, shell=True)

# for v in ['linear','constant','exponential','step']:
#     cmd_str = f'python plot.py --title train loss vs epoch for {v} --files ./../experiments/part_1_1/sgd_{v}/log.txt ./../experiments/part_1_1/sgd_mom_{v}/log.txt ./../experiments/part_1_1/rmsprop_{v}/log.txt ./../experiments/part_1_1/adam_{v}/log.txt ./../experiments/part_1_1/adagrad_{v}/log.txt --legends sgd sgd_mom rmsprop adam adagrad --column 0 --file_name plots/{v}.png'
#     subprocess.run(cmd_str, shell=True)

# for v in ['linear','constant','exponential','step']:
#     cmd_str = f'python plot.py --title train loss vs epoch for {v} --files ./../experiments/early_stop/lr_01/sgd_{v}/log.txt ./../experiments/early_stop/lr_01/sgd_mom_{v}/log.txt ./../experiments/early_stop/lr_01/rmsprop_{v}/log.txt ./../experiments/early_stop/lr_01/adam_{v}/log.txt ./../experiments/early_stop/lr_01/adagrad_{v}/log.txt --legends sgd sgd_mom rmsprop adam adagrad --column 0 --file_name plots/{v}_lr_01.png'
#     subprocess.run(cmd_str, shell=True)


# for v in ['linear','constant','exponential','step']:
#     for o in ['adam','adagrad','sgd','rmsprop', 'sgd_mom']:
#         cmd_str = f'python plot.py --title loss vs epoch for {o} {v} --files ./../experiments/early_stop/lr_01/{o}_{v}/log.txt ./../experiments/early_stop/lr_01/{o}_{v}/log.txt --legends train val --column 0 1 --file_name plots/{o}_{v}_early_stop_lr_01.png'
#         subprocess.run(cmd_str, shell=True)

# for p in range(1,4):
#     for v in ['linear','exponential']:
#         for o in ['adagrad','sgd','sgd_mom']:
#             cmd_str = f'python plot.py --title loss vs epoch for {o} {v} patience {p} --files ./../experiment/early_stop/patience_{p}/{o}_{v}/log.txt ./../experiment/early_stop/patience_{p}/{o}_{v}/log.txt --legends train val --column 0 1 --file_name plots/{o}_{v}_early_stop_patience_{p}.png'
#             subprocess.run(cmd_str, shell=True)
        
optimizer = {}
optimizer['adagrad'] = 'Adagrad'
optimizer['adam'] = 'Adam'
optimizer['rmsprop'] = 'RMSprop'
optimizer['sgd'] = 'SGD'

# for p in range(1,6):
#     for v in ['linear','exponential', 'step', 'const']:
#         for o in ['adagrad','sgd','rmsprop', 'adam']:
#             print(p, v, o)
#             cmd_str = f'python main.py --task train --args experiment_name {o}_{v} root ./../experiment/early_stop/patience_{p} optimizer {optimizer[o]} lr 0.1 patience {p} lr_scheduler.name {v} val_split 0.2'
#             subprocess.run(cmd_str, shell=True)
#         cmd_str = f'python main.py --task train --args momentum 0.9 experiment_name sgd_mom_{v} root ./../experiment/early_stop/patience_{p} lr 0.1 patience {p} lr_scheduler.name {v} val_split 0.2'
#         subprocess.run(cmd_str, shell=True)
        
# r = []
# for p in range(1,4):
#     r = []
#     for n in ['bn', 'bin']:
#             v = 'exponential'
#             o = 'adagrad'
#             name = f'{n}-P{p} &'
#             t = get_config(['resume_dir', f'./../experiment/part_1_2/early_stop/patience_{p}/{o}_{v}_{n}'])
#             t.normalization = n
#             tester = Tester(t)
#             result = tester.test()
#             result = result + " \\\\"
#             r.append(name + result)
#     for s in r:
#         print(s)

            # print('test')
            # cmd_str = f'python main.py --task test --args resume_dir ./../experiment/early_stop/patience_{p}/{o}_{v}'       
            # subprocess.run(cmd_str, shell=True)
            # print()

# for p in range(1,4):
#     for n in ['inbuilt', 'ln', 'bn', 'gn','in','bin', 'none']:
#         o = 'adagrad'
#         v = 'exponential'
#         cmd_str =  f'python plot.py --title loss vs epoch for {n} --files ./../experiment/part_1_2/early_stop/patience_{p}/{o}_{v}_{n}/log.txt ./../experiment/part_1_2/early_stop/patience_{p}/{o}_{v}_{n}/log.txt --legends train val --column 0 1 --file_name plots/{o}_{v}_patience_{p}_{n}.png'
#         subprocess.run(cmd_str, shell=True)



# for n in ['inbuilt', 'ln', 'bn', 'gn','in','bin', 'none']:
#     cmd_str =  f'python main.py --task quantile --args normalization {n} root ./../experiment/part_1_2_7/ experiment_name adagrad_exponential_{n} optimizer Adagrad lr_scheduler.name exponential val_split 0.2 epochs 30'
#     print(cmd_str)
#     subprocess.run(cmd_str, shell=True)

for n in ['inbuilt', 'ln', 'bn', 'gn','in','bin', 'none']:
    cmd_str =  f'python plot.py --title quantile vs epoch for {n} --files ./../experiment/part_1_2_7/adagrad_exponential_{n}/quatile.txt ./../experiment/part_1_2_7/adagrad_exponential_{n}/quatile.txt ./../experiment/part_1_2_7/adagrad_exponential_{n}/quatile.txt ./../experiment/part_1_2_7/adagrad_exponential_{n}/quatile.txt --column 0 1 2 3  --legends 1 20 80 90 --file_name plots/{n}_quantile.png'
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)



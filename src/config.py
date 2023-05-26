from yacs.config import CfgNode

default_config = {
    'r': 10,
    'n': 2,
    'normalization': 'inbuilt',
    'data': ['./../data/cifar-10-batches-py/data_batch_1',
            './../data/cifar-10-batches-py/data_batch_2',
            './../data/cifar-10-batches-py/data_batch_3',
            './../data/cifar-10-batches-py/data_batch_4',
            './../data/cifar-10-batches-py/data_batch_5'] ,

    'test_data': ['./../data/cifar-10-batches-py/test_batch'],
    
    'experiment_name': 'demo',
    'root': './../experiments/part_1_1',
    'resume_dir':None,
    'output_file':None,
    
    'epochs':105,
    'lr': 0.1,
    'device':'cuda',
    'checkpoint_every':25,
    'checkpoint': None,
    'test_on_train': False,

    'train':{
        'batch_size':128
    },
    'test':{
        'batch_size':128
    },
    'val_split': None,

    'optimizer':'SGD',
    'momentum': 0.0,
    'patience': None,
    
    'lr_scheduler':{
        'name': 'const',
        'gamma': 0.99,
        'start_factor':0.5,
        'end_factor':1,
        'total_iters': 50,
        'step_size':20,
        'factor':0.8
    }
}
default_config = CfgNode(default_config)

#get the config from the extras
def get_config(extra_args):
    default_config.set_new_allowed(True)
    default_config.merge_from_list(extra_args)
    default_config.extras = extra_args
    return default_config


#load from a file
def load_config(file_name, new_config):
    
    #the file of args
    new_config.set_new_allowed(True)
    extras = new_config.extras
    new_config.merge_from_file(file_name)
    new_config.merge_from_list(extras)
    new_config.extras = []
    return new_config

#dump to file
def dump_to_file(file_name, config):
    config_str = config.dump()
    with open(file_name, "w") as f:
        f.write(config_str)
    
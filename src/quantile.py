from model_q import ResNet
from dataloader import get_dataloader
from torchsummary import summary
from tqdm import tqdm 
from torch.nn.functional import one_hot
from torch.optim import SGD, Adam, RMSprop, Adagrad
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, LinearLR, StepLR
from config import dump_to_file

import pdb
import torch
import os
import sys

'''
Info regarding schedulers
1. ConstantLR: keep lr = lr_0*factor till totoal iter then lr = lr_0
2. ExponentialLR: lr = gamma*lr
'''

class Quantile:

    def __init__(self,args):

        #store the args
        self.args = args

        #get the meta data
        self.epochs = self.args.epochs
        self.lr = self.args.lr
        self.device = self.args.device
        self.init_dirs()
        
        #validation and checkpoint frequency
        self.validate_every = 1
        if(self.args.val_split is None):
            self.validate_every = 1000000
        self.checkpoint_every = self.args.checkpoint_every
        
        #for early stop patience
        self.patience = self.args.patience
        if(self.patience is None):
            self.patience = self.epochs + 1
        self.bad_validation = 0
        
        #get the data sets
        if(self.args.val_split is not None):
            self.val_data, self.train_data = get_dataloader(args, mode = 'train')
        else:
            self.train_data = get_dataloader(args, mode = 'train')

        #get the model
        self.model = ResNet(self.args.n, self.args.r, self.args.normalization).to(self.device)#, normalization)
        
        #get the optimizer
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()
        
        #the minimum validation loss
        self.min_validation_loss = sys.float_info.max 

    def init_dirs(self):

        #create the root dirs
        self.root = os.path.join(self.args.root, self.args.experiment_name)
        if(not os.path.exists(self.root)):
            os.makedirs(self.root)
        self.checkpoint_dir = os.path.join(self.root, 'checkpoint')
        if(not os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
        self.log_file = os.path.join(self.root, 'log.txt')
        with open(self.log_file, "w") as f:
            f.write('train_loss, val_loss\n')
            pass
        self.args_file = os.path.join(self.root, 'config.yaml')
        
        self.quantile_file = os.path.join(self.root, 'quatile.txt')
        with open(self.quantile_file, "w") as f:
            pass
        
        dump_to_file(self.args_file, self.args)
        
    def get_optimizer(self):
        rhos, params = self.get_parameters()
        params = [{"params": rhos, "lr": self.lr*10, "weight_decay":0.0},{'params': params}]
        if (not 'optimizer' in self.args
            or self.args.optimizer == 'SGD'):
            return SGD(params, lr = self.lr, weight_decay=0.0001, momentum = self.args.momentum)
        elif(self.args.optimizer == 'Adam'):
            return Adam(params, lr = self.lr, weight_decay=0.0001)
        elif(self.args.optimizer == 'RMSprop'):
            return RMSprop(params, lr = self.lr, weight_decay=0.0001)
        elif(self.args.optimizer == 'Adagrad'):
            return Adagrad(params, lr = self.lr, weight_decay=0.0001)
            
    def get_parameters(self):
        rhos = []
        non_rhos = []
        for name, param in self.model.named_parameters():
            if('rho' in name):
                rhos.append(param)
            else:
                non_rhos.append(param)
        return rhos, non_rhos
    '''
    Assume lr is 1 in the start
    2. ExponentialLR: (gamms = 0.95)
        1, 1*(0.95), 1*(0.95)^2, 1*(0.95)^3, 1*(0.95)^4, 1*(0.95)^5
    3. LinearLR: (iters = 5, start_factor = 0.5, end_factor = 1)
        1*0.5, 1*0.6, 1*0.7, 1*0.8, 1*0.9, 1, 1, 1
    4. StepLR: (step_size = 5, factor = 0.1)
        1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01
    '''

    def get_lr_scheduler(self):    
        if (self.args.lr_scheduler.name == 'const'):
            return ConstantLR(self.optimizer, factor = 1.)

        elif(self.args.lr_scheduler.name == 'exponential'):
            return ExponentialLR(self.optimizer, 
                gamma = self.args.lr_scheduler.gamma)
        
        elif(self.args.lr_scheduler.name == 'linear'):
            return LinearLR(self.optimizer, 
                start_factor = self.args.lr_scheduler.start_factor, 
                end_factor = self.args.lr_scheduler.end_factor,
                total_iters = self.args.lr_scheduler.total_iters)
        
        elif(self.args.lr_scheduler.name == 'step'):
            return StepLR(self.optimizer, 
                gamma = self.args.lr_scheduler.factor,
                step_size = self.args.lr_scheduler.step_size)

    def save_model(self, best = False):
        if(best):
            checkpoint_name = f'best.pt'
        else:
            checkpoint_name = f'model_{self.epoch_no}.pt'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def save_results(self, results):
        with open(self.log_file, "a") as f:
            s = ''
            for loss in results:
                s += str(loss) + ','
            f.write(f'{s}\n')
            f.close()

    def print_results(self, results, validate = False):
        
        #print type of results
        print()
        if(validate):
            print("----VALIDITAION----")
        else:
            print(f'----EPOCH: {self.epoch_no}----')
        
        #print losses
        print(f'loss: {results}')
        print()
        print("-"*20)
    

    def loss(self, true_label, predicted_labels):
        #do one-hot mapping of true labels
        #adding some small offset to avoid 0
        predicted_labels = (predicted_labels + 1e-15)
        true_label = one_hot(true_label, num_classes = self.args.r)
        loss = -1*true_label*torch.log(predicted_labels)
        return (loss.sum(-1)).mean()

    #validate function is just same as the train 
    def validate(self):
        
        #we need to return the average loss
        losses = []

        #with torch no grad
        with torch.no_grad():
            self.model.eval()

            results = []
        
            with tqdm(total=len(self.val_data)) as t:   

                for (data, true_label) in self.val_data:
                    
                    #zero out the gradient
                    self.optimizer.zero_grad()

                    #predict the labels and loss
                    _ , f = self.model(data)
                    results.append(f)
                
                #final tensor
                t = torch.cat(results, dim = 0).flatten().sort()[0]
                l = len(t)
                y = [t[int(l*0.01)].item(), t[int(l*0.2)].item(), t[int(l*0.8)].item(), t[int(l*0.99)].item()]
                return y

    def train_epoch(self):
        
        #we need to return the average loss
        losses = []

        #set in the train mode
        self.model.train()

        #run for batch
        with tqdm(total=len(self.train_data)) as t:   

            for (data, true_label) in self.train_data:
                
                #zero out the gradient
                self.optimizer.zero_grad()

                #predict the labels and loss
                predicted_labels, _ = self.model(data)
                loss = self.loss(true_label, predicted_labels)

                #do backpropogation
                loss.backward()
                self.optimizer.step()

                #append loss to the file
                losses.append(loss.item())

                #update the progress bar
                t.set_postfix(loss=f"{loss:.2f}")
                t.update(1)     
            
        return sum(losses)/len(losses)

    def quantile(self):
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch_no = epoch

            #the list which goes in the log text
            log = []
            
            #train for one epoch and print results
            train_loss = self.train_epoch()
            self.print_results(train_loss)
            log.append(train_loss)

            #change lr scheduler
            self.lr_scheduler.step()

            # #checkpoint the model
            # if((epoch + 1)%self.checkpoint_every == 0):
            #     self.save_model()

            #do validation if neccesary
            if((epoch + 1)% self.validate_every == 0):
                r = self.validate()
                with open(self.quantile_file,'a') as f:
                    for p in r:
                        f.write(str(p) + ',')
                    f.write('\n')
                    
                
            #write results to the file
            self.save_results(log)

        print(f'min_validation_loss: {self.min_validation_loss}')
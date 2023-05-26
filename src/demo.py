import pickle
import pdb
import torch 
import numpy as np 
from torch.utils.data import DataLoader, random_split
import csv
class Data:

    def __init__(self, args, data_paths, mode = 'train'):

        #read the files
        self.args = args
        self.data = []
        self.labels = []
        self.device = self.args.device

        for file_path in data_paths:
            
            #dictionary containgin
            #b'data', b'labels'
            batch_data = self.unpickle(file_path)
            y = open('data.csv', 'w')
            y = csv.writer(y)
            u = open('gold.txt', 'w')
            for i in range(0,100):
                y.writerow(batch_data[b'data'][i])
                u.write(str(batch_data[b'labels'][i]) + '\n')
                
            break


            batch_data[b'data'] = torch.tensor(batch_data[b'data'].reshape(-1,3,32,32)).float()
            
            self.data.append(batch_data[b'data'])
            self.labels.append(batch_data[b'labels'])

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __getitem__(self, index):
        batch = index // 10000
        index = index % 10000

        data = self.data[batch][index].to(self.device)
        
        data = (data/255 - self.mean)/self.std
        label = torch.tensor(self.labels[batch][index]).to(self.device)
        return data, label

    def __len__(self):
        return sum([len(d) for d in self.data])

'''
returns dataloader depending on mode
if mode == train:
    if val_split is not none:
        train, val
    else:
        train
else:
    test
'''
# #the collat function
# def collate_fn_help(device, batch):
#     pdb.set_trace()
#     data = batch[0].to(device)
#     labels = batch[1].to(device)
#     return data, labels


from config import get_config
args = get_config([])
t = Data(args,args.data)
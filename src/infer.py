#python3 infer.py --model_file models/lstm_lstm_attn.pth --model_type lstm_lstm_attn --test_data_file sql_query.csv --output_file output.csv

from argparse import ArgumentParser
import torch
import csv 
from tqdm import tqdm
from model import ResNet
import pdb
import numpy as np
def get_parser():
    
    #the parser for the arguments
    parser = ArgumentParser(
                        prog = 'python main.py',
                        description = 'This is main file for COL775 A1. You can train model from scratch or resume training, test checkpoint from this file',
                        epilog = 'thank you!!')

    #there are two tasks ['train', 'test']
    parser.add_argument('--model_file', default = '', required=True)

    #there are two tasks ['train', 'test']
    parser.add_argument('--normalization',  choices=['bn','in','bin','ln','gn','nn','inbuilt'], required=True)
    parser.add_argument('--n', type = int, default = 2)
    #files
    parser.add_argument('--test_data_file', default = '', required=True)
    parser.add_argument('--output_file', default = '', required=True)
    
    return parser

def solve_for_one_lstm(model, text_tokens, value_mapping, db):
    
    tokens = [model.encoder_vocab(i) for i in text_tokens]
    tokens = torch.tensor(tokens).unsqueeze(0).cuda()
    
    db_id = torch.tensor([model.decoder.dbid_dict[db]]).cuda()
    
    output = model(db_id, tokens)[1:]
    for i in range(len(output)):
        output[i] = model.decoder.embeddings.vocab_inv[output[i]]
    output = (' '.join(output))
    output = output.replace(' . ', '.')
    output = output.replace('. ', '.')
    output = output.replace(' .', '.')
    output = output.replace('<EOS>', '')
    for key in value_mapping:
        output = output.replace(value_mapping[key], key)
    return output



def solve_for_one_bert(model, text_tokens, value_mapping, db):
    
    db_id = torch.tensor([model.decoder.dbid_dict[db]]).cuda()
    text_tokens = ' '.join(text_tokens)
    output = model(db_id,[text_tokens])[1:]
    for i in range(len(output)):
        output[i] = model.decoder.embeddings.vocab_inv[output[i]]
    output = (' '.join(output))
    output = output.replace(' . ', '.')
    output = output.replace('. ', '.')
    output = output.replace(' .', '.')
    output = output.replace('<EOS>', '')
    for key in value_mapping:
        output = output.replace(value_mapping[key], key)
    return output
# class CustomUnpickler(pickle.Unpickler):

#     def find_class(self, module, name):
#         if name == 'Manager':
#             from settings import Manager
#             return Manager
#         return super().find_class(module, name)


if __name__ == '__main__':

    #parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    
    model = ResNet(args.n, 10, args.normalization).cuda()
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    
    #get the parser
    
    #read file one by one
    data_file = open(args.test_data_file, 'r')
    data_file = csv.reader(data_file)
    
    #output file
    # outfile = csv.writer(open(args.output_file, 'w'))
    # goldfile = csv.writer(open('gold.csv', 'w'))

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).unsqueeze(-1).unsqueeze(-1).cuda()
    std = torch.tensor([0.247, 0.243, 0.261]).unsqueeze(-1).unsqueeze(-1).cuda()

    outfile = open(args.output_file, 'w')

    for line in tqdm(data_file):
        t = np.array([int(i) for i in line])
        data = torch.tensor(t.reshape(-1,3,32,32)).float().cuda()
        data = (data/255 - mean)/std
        out = torch.argmax(model(data)).item()
        outfile.write(str(out) + "\n")

        

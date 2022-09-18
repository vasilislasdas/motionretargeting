import torch
import torch.utils.data as data
import torch
import torch.nn as nn
import sys
sys.path.append( '../models')
from transformer_models import TransformerEncoderDecoder
import torchinfo
import tqdm



class ReverseSequences( data.Dataset ):

    def __init__(self, seq_len, dataset_size ):
        super(ReverseSequences,self).__init__()
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.generate_sequence()

    def generate_sequence(self):
        x = torch.randint(low=0,high=10,size=(self.dataset_size, self.seq_len) ).float()
        y = torch.flip(x, dims=[1])
        self.x = x
        self.y = y

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        return( self.x[ item ], self.y[item])


class ReverseSequenceSummed( data.Dataset ):

    def __init__( self, seq_len, dims,  dataset_size ):
        super(ReverseSequenceSummed,self).__init__()
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.dims = dims
        self.generate_sequence()

    def generate_sequence(self):
        x = torch.randint(low=0,high=10,size=( self.dataset_size, self.seq_len,self.dims ) ).float()
        y = torch.flip(x, dims=[1])
        y = torch.cumsum( y, dim = 1 )
        self.x = x
        self.y = y

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        return( self.x[ item ], self.y[item])



if __name__ == "__main__":
    d = ReverseSequenceSummed(seq_len=7, dims=2, dataset_size=10)
    print(d[0])

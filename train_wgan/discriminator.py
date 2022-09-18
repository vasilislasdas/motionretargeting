import torch
import torch.nn as nn
from transformerlayer import EncoderLayer, DecoderLayer,PositionalEncoding
import math
import sys
sys.path.append( '../utils')
# from utils.datasetRetargeting import  RetargetingDataset
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from utils import *
import einops



class Discriminator( nn.Module ):

    def __init__(self, nr_layers: int,
                       dim_input: int,
                       dim_model: int,
                       seq_len: int,
                       dim_skeleton: int,
                       nr_heads: int,
                       nr_classes: int,
                       dropout : float = 0.1,
                       dim_inter: int = None ):

        super( Discriminator, self ).__init__()

        self.nr_layers = nr_layers
        self.dim_input = dim_input # dimensionality of each frame in the input sequence
        self.dim_model = dim_model # internal dim of encoder
        self.seq_len = seq_len # sequence length: fixed for my case
        self.skeleton_dim = dim_skeleton
        self.nr_heads = nr_heads
        self.dropout= dropout
        self.dim_inter = dim_inter
        self.nr_class = nr_classes

        # projects the input to the internal dimensionality of the model: initially i concatenate skeleton+motion data
        self.linear_input = nn.Linear( self.dim_input, self.dim_model )

        # modified wasserstein output: for each frame provide a number..real or fake
        # self.classes = nn.Linear( self.dim_model*self.seq_len, self.seq_len )
        self.frames = nn.Linear(self.dim_model, 1)
        self.diff_frames = nn.Linear(self.dim_model, 1)


        # prepend to the motion data an embedding of the skeleton
        # self.embed_skeleton = nn.Linear( self.skeleton_dim, self.dim_model )

        # positional embeddings
        self.positions = PositionalEncoding( d_model=self.dim_model )


        # encoder
        self.encoder = nn.ModuleList( [ EncoderLayer( dim_model=self.dim_model,
                                                      nr_heads=self.nr_heads,
                                                      dropout=self.dropout,
                                                      dim_inter=self.dim_inter ) for _ in range( self.nr_layers) ] )


    def forward( self, input_seq:torch.Tensor, input_skeleton: torch.Tensor):

        assert (torch.any(torch.isnan(input_seq)) == False )
        assert (torch.any(torch.isnan(input_skeleton)) == False )
        assert (torch.any(torch.isinf(input_seq)) == False)
        assert (torch.any(torch.isinf(input_skeleton)) == False)

        # make sure we are given 3D-tensors for the encoder and decoder: batch_size X seq_len X input_dim
        assert(input_seq.ndim == 3 )

        # project input sequence dim to encoder_dim
        input_seq = self.linear_input(input_seq)

        # add positional embeddings to the embeddings
        input_seq = self.positions(input_seq)

        # compute encoder representations
        encoder_reps = input_seq
        for encoder_layer in self.encoder:
            encoder_reps, encoder_attention = encoder_layer( encoder_reps )

        # difference between adjacent frames
        diff_reps = encoder_reps[:,1:,:] - encoder_reps[:,0:-1,:]

        # classify also frame-diff
        out_frames = self.frames(encoder_reps).squeeze()
        out_diff_frames = self.diff_frames(diff_reps).squeeze()


        return out_frames,out_diff_frames



if __name__ == "__main__":
    print("TESTING DISCRIMINATOR    ")

    batch_size = 111
    nr_classes = 6

    network = Discriminator(nr_layers =  4,
                       dim_input = 111,
                       dim_model =  128,
                       seq_len = 30,
                       dim_skeleton = 135,
                       nr_classes = nr_classes,
                       nr_heads = 8,
                       dropout = 0.1 )

    # fake input data
    motion1 = torch.rand(batch_size,30,111)
    skeleton1 = torch.rand(batch_size, 1, 135)


    # fake labels
    labels = torch.empty( batch_size, dtype=torch.long).random_(nr_classes)
    print(f"labels:{labels}")

    # loss function
    loss_fn = nn.CrossEntropyLoss()  # average cross entropy loss over each prefix in batch

    print(f"motion,skeleton shape:{motion1.shape,skeleton1.shape}")
    out = network(motion1, skeleton1)
    loss = loss_fn(out, labels)
    print(f"output shape:{out.shape}")


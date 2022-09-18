import torch
import torch.nn as nn
from transfomerlayer import EncoderLayer, DecoderLayer,PositionalEncoding
import math
import sys
sys.path.append( '../utils')
# from utils.datasetRetargeting import  RetargetingDataset
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from utils import *
import einops



class RetargeterGeneratorEncoder( nn.Module ):

    def __init__(self, nr_layers: int,
                       dim_input: int,
                       dim_model: int,
                       seq_len: int,
                       dim_skeleton: int,
                       nr_heads: int,
                       dropout : float = 0.1,
                       dim_inter: int = None ):

        super( RetargeterGeneratorEncoder, self ).__init__()

        self.nr_layers = nr_layers
        self.dim_input = dim_input # dimensionality of each frame in the input sequence
        self.dim_model = dim_model # internal dim of encoder
        self.seq_len = seq_len # sequence length: fixed for my case
        self.skeleton_dim = dim_skeleton
        self.nr_heads = nr_heads
        self.dropout= dropout
        self.dim_inter = dim_inter

        # projects the input to the internal dimensionality of the model: initially i concatenate skeleton+motion data
        self.linear_input = nn.Linear( self.dim_input+self.skeleton_dim, self.dim_model )

        # projects the flattened output of the transformer-encoder to a flattened motion clip
        # NOTE: since the target skeleton is prepended to the input seq it is +1 internally
        # self.linear_output = nn.Linear( self.dim_model*(seq_len), self.dim_input*seq_len )
        self.linear_output = nn.Linear( self.dim_model, self.dim_input )
        self.linear_output2 = nn.Linear( self.dim_input, self.dim_input)
        self.relu = nn.LeakyReLU(0.1)
        # self.relu = nn.ReLU()

        # prepend to the motion data an embedding of the skeleton
        self.embed_skeleton = nn.Linear( self.skeleton_dim, self.dim_model )
        # self.embed_skeleton.requires_grad_(False)

        # positional embeddings
        self.positions = PositionalEncoding( d_model=self.dim_model )


        # encoder
        self.encoder = nn.ModuleList( [ EncoderLayer( dim_model=self.dim_model,
                                                      nr_heads=self.nr_heads,
                                                      dropout=self.dropout,
                                                      dim_inter=self.dim_inter ) for _ in range( self.nr_layers) ] )


    def forward( self, input_seq:torch.Tensor,
                       input_skeleton: torch.Tensor,
                       target_skeleton:torch.tensor ):

        # print("HERE!!")
        assert (torch.any(torch.isnan(input_seq)) == False )
        assert (torch.any(torch.isnan(input_skeleton)) == False )
        assert (torch.any(torch.isnan(target_skeleton)) == False )
        assert (torch.any(torch.isinf(input_seq)) == False)
        assert (torch.any(torch.isinf(input_skeleton)) == False)
        assert (torch.any(torch.isinf(target_skeleton)) == False)

        # make sure we are given 3D-tensors for the encoder and decoder: batch_size X seq_len X input_dim
        assert(input_seq.ndim == 3 )

        # concatenate input skeleton with the motion data
        input_skeleton = einops.repeat( input_skeleton, "batch 1 dim -> batch seq_len dim", seq_len = input_seq.shape[1] )
        input_seq = torch.cat( [ input_seq,input_skeleton], dim=2 )
        # print(input_seq.shape)

        # project input sequence dim to encoder_dim
        input_seq = self.linear_input(input_seq)

        # add positional embeddings to the embeddings
        input_seq = self.positions(input_seq)

        # project target skeleton to encoder_dim for concatenation
        # print(f"Target skeleton:{target_skeleton.shape}")
        target_skeleton = self.embed_skeleton( target_skeleton )

        # prepend target_skeleton embdedding to the input embeddings
        input_seq = torch.cat( [ target_skeleton, input_seq  ], dim = 1 )
        # print(f"Concatenated Input seq:{input_seq.shape}")

        # compute encoder representations
        encoder_reps = input_seq
        for encoder_layer in self.encoder:
            encoder_reps, encoder_attention = encoder_layer( encoder_reps )

        # pass the encoder representation through a simple linear layer to fetch the new motion output
        # print(f"encoder{encoder_reps.shape}")
        encoder_reps = encoder_reps[:, 1:, :]
        # print(f"encoder{encoder_reps.shape}")

        # encoder_output = self.linear_output2(  self.relu( self.linear_output( encoder_reps.view( encoder_reps.shape[0], -1) ) ) )
        # encoder_output = self.linear_output( encoder_reps.view( encoder_reps.shape[0], -1) )
        encoder_output = self.linear_output2( self.relu( self.linear_output(encoder_reps)))
        # encoder_output = self.linear_output(encoder_reps)

        #  restore the encoder output motion dims
        #  note: the linear layer above took into account all frames: not only eacg frame seperetaly
        # encoder_output = encoder_output.reshape( encoder_output.shape[0], self.seq_len, self.dim_input )
        # print( f"Encoder output dimensions:{encoder_output.shape}")

        # return everything
        return( encoder_output, encoder_reps, encoder_attention )





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



class RetargeterGeneratorEncoderBoth( nn.Module ):

    def __init__(self, nr_layers: int,
                       dim_motion1: int,
                       dim_motion2: int,
                       dim_model: int,
                       seq_len: int,
                       dim_skeleton1: int,
                       dim_skeleton2: int,
                       nr_heads: int,
                       dropout : float = 0.1,
                       dim_inter: int = None ):

        super( RetargeterGeneratorEncoderBoth, self ).__init__()

        self.nr_layers = nr_layers

        # each motion has different dims
        self.dim_motion1 = dim_motion1  # dim of first type of skeletons
        self.dim_motion2 = dim_motion2  # dim of second type of skeletons

        # each skeleton has different dims
        self.skeleton_dim1 = dim_skeleton1
        self.skeleton_dim2 = dim_skeleton2

        # decoder dimensionality
        self.dim_model = dim_model # internal dim of encoder: fixed and shared for both skeletons
        self.seq_len = seq_len # sequence length: fixed for my case

        # heads, dropout, internal dim of encoder
        self.nr_heads = nr_heads
        self.dropout= dropout
        self.dim_inter = dim_inter

        # projects the souce motion+ source skeleton to the internal dimensionality of the model
        self.linear_input1 = nn.Linear( self.dim_motion1 + self.skeleton_dim1, self.dim_model )
        self.linear_input2 = nn.Linear( self.dim_motion2 + self.skeleton_dim2, self.dim_model)

        # prepend to the motion data an embedding of the skeletons
        self.embed_skeleton1 = nn.Linear( self.skeleton_dim1, self.dim_model )
        self.embed_skeleton2 = nn.Linear( self.skeleton_dim2, self.dim_model )



        ## encoder output goes to 2 feed-forward nets acting as decoders
        # used by all
        self.relu = nn.LeakyReLU(0.1)

        # decoder 1
        self.linear_ffn11 = nn.Linear( self.dim_model, self.dim_motion1 )
        self.linear_ffn12 = nn.Linear( self.dim_motion1, self.dim_motion1)
        self.linear_output1 = nn.Linear(  self.dim_motion1, self.dim_motion1)

        # decoder 2
        self.linear_ffn21 = nn.Linear(self.dim_model, self.dim_motion2)
        self.linear_ffn22 = nn.Linear( self.dim_motion2, self.dim_motion2)
        self.linear_output2 = nn.Linear( self.dim_motion2, self.dim_motion2 )

        # positional embeddings for the (concatenated) motion only
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
        # print(f"input_seq:{input_seq.shape}")

        # project input sequence dim to encoder_dim and use the correct linear projection
        if input_seq.shape[-1] == (self.dim_motion1 + self.skeleton_dim1):
            input_seq = self.linear_input1(input_seq)
        elif input_seq.shape[-1] == ( self.dim_motion2 + self.skeleton_dim2 ):
            input_seq = self.linear_input2(input_seq)
        else:
            assert( False )


        # add positional embeddings to the embedded motion+skeletal data
        input_seq = self.positions( input_seq )

        # project target skeleton to encoder_dim for concatenation
        # print(f"Target skeleton:{target_skeleton.shape}")
        target_skeleton_dims = target_skeleton.shape[-1]
        if target_skeleton_dims == self.skeleton_dim1:
            target_skeleton = self.embed_skeleton1( target_skeleton )
        elif target_skeleton_dims == self.skeleton_dim2:
            target_skeleton = self.embed_skeleton2( target_skeleton )
        else:
            assert(False )
        # print(f"Target skeleton:{target_skeleton.shape}")


        # prepend target_skeleton embdedding to the input embeddings
        input_seq = torch.cat( [ target_skeleton, input_seq  ], dim = 1 )
        # print(f"Concatenated Input seq:{input_seq.shape}")

        # compute encoder representations
        encoder_reps = input_seq
        for encoder_layer in self.encoder:
            encoder_reps, encoder_attention = encoder_layer( encoder_reps )

        # pass the encoder through an ffd network: ditch the first element which corresponds to the target skeleton
        # print(f"encoder{encoder_reps.shape}")
        encoder_reps = encoder_reps[:, 1:, :]
        # print(f"encoder{encoder_reps.shape}")

        # go back to the original dimensions
        if target_skeleton_dims == self.skeleton_dim1:
            # encoder_output = self.linear_output1( self.relu( self.linear_ffn12( self.relu( self.linear_ffn11( encoder_reps ) ) ) ) )
            encoder_output = self.linear_output1( self.relu( self.linear_ffn11(encoder_reps)))
        elif target_skeleton_dims == self.skeleton_dim2:
            # encoder_output = self.linear_output2( self.relu( self.linear_ffn22( self.relu( self.linear_ffn21( encoder_reps ) ) ) ) )
            encoder_output = self.linear_output2( self.relu( self.linear_ffn21(encoder_reps)))
        else:
            assert( False )


        # return everything
        return( encoder_output, encoder_reps, encoder_attention )


if __name__ == "__main__":
    print("TESTING NEW MODEL    ")
    network = RetargeterGeneratorEncoderBoth(nr_layers =  4,
                       dim_motion1 = 111,
                       dim_motion2 =  91,
                       dim_model =  128,
                       seq_len = 30,
                       dim_skeleton1 = 135,
                       dim_skeleton2 = 110,
                       nr_heads = 8,
                       dropout = 0.1 )

    motion1 = torch.rand(1,30,111)
    motion2 = torch.rand(1, 30, 91)
    skeleton1 = torch.rand(1, 1, 135)
    skeleton2 = torch.rand(1, 1, 110)

    out,_,_ = network(motion1, skeleton1, skeleton1)
    print(f"output shape:{out.shape}")

    out, _, _ = network(motion1, skeleton1, skeleton2)
    print(f"output shape:{out.shape}")

    out, _, _ = network(motion2, skeleton2, skeleton2)
    print(f"output shape:{out.shape}")

    out, _, _ = network(motion2, skeleton2, skeleton1)
    print(f"output shape:{out.shape}")
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



class RetargeterGenerator( nn.Module ):

    def __init__(self, nr_layers:int,
                       dim_input:int,
                       dim_model: int,
                       dim_output:int = None,
                       nr_heads: int = 1,
                       dropout : float = 0.1,
                       dim_inter: int = None ):

        super(RetargeterGenerator, self).__init__()

        self.nr_layers = nr_layers
        self.dim_input = dim_input # dimensionality of each symbol in the input sequence
        self.dim_model = dim_model # internal dim for both encoder decoder
        self.nr_heads = nr_heads
        self.dropout= dropout
        self.dim_inter = dim_inter
        self.dim_output = dim_output # dimensionality of each symbol in the output sequence

        # projects the input to the internal dimensionality of the model
        self.linear_input = nn.Linear( self.dim_input, self.dim_model )

        # used in the decoder to specify a skeleton-class: used for the conditioning-part
        self.class_embedding = nn.Embedding( 25, self.dim_model )
        # positional embeddings
        self.positions = PositionalEncoding(d_model=self.dim_model)


        # projects the output of the transformers to the desired dimenionsnality
        if self.dim_output is None:
            self.dim_output = dim_input
        self.linear_output = nn.Linear( self.dim_model, self.dim_output )
        self.linear_output2 = nn.Linear( self.dim_output, self.dim_output)
        # self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

        # encoder
        self.encoder = nn.ModuleList( [ EncoderLayer( dim_model=self.dim_model,
                                                      nr_heads=self.nr_heads,
                                                      dropout=self.dropout,
                                                      dim_inter=self.dim_inter ) for _ in range( self.nr_layers) ] )

        # decoder
        self.decoder = nn.ModuleList( [ DecoderLayer( dim_model=self.dim_model,
                                                      nr_heads= self.nr_heads,
                                                      dropout=self.dropout,
                                                      dim_inter=self.dim_inter ) for _ in range( self.nr_layers) ] )


    def forward( self, input_seq:torch.Tensor,
                       target_seq:torch.Tensor,
                       input_skeleton,
                       target_skeleton,
                       class_in,
                       class_out):

        # make sure we are given 3D-tensors for the encoder and decoder: batch_size X seq_len X input_dim
        assert(input_seq.ndim == 3 and target_seq.ndim == 3 )
        # print(f"Input seq:{input_seq.shape}")
        # print(f"Output seq:{target_seq.shape}")

        # repeat skeleton for every frame in input and target_sequences
        # print(f"Input skeleton seq:{input_skeleton.shape}")
        # print(f"Output skeleton seq:{target_skeleton.shape}")
        nr1 = input_seq.shape[1]
        nr2 = target_seq.shape[1]
        input_skeleton = einops.repeat(  input_skeleton, 'm 1 n -> m k n', k=nr1)
        target_skeleton = einops.repeat( target_skeleton, 'm 1 n -> m k n', k=nr2)
        # print(f"Input skeleton seq:{input_skeleton.shape}")
        # print(f"Output skeleton seq:{target_skeleton.shape}")

        # concatenate skeleton + input before the embedding
        input_seq = torch.cat( [ input_seq, input_skeleton ], dim = 2 )
        target_seq = torch.cat([ target_seq, target_skeleton ], dim = 2)
        # print(f"Concatenated Input seq:{input_seq.shape}")
        # print(f"Concatenated Output seq:{target_seq.shape}")

        # first project the dimensionality of the input to the internal dim of the encoder-decoder blocks
        input_seq =  self.linear_input( input_seq )
        target_seq = self.linear_input(target_seq)


        # add positional embeddings
        input_seq = self.positions(input_seq)
        target_seq = self.positions(target_seq)

        # get class number and fetch embedding for the skeleton-class
        # print( class_in )
        # print(class_out)
        class_embed_in = self.class_embedding(class_in).unsqueeze(dim=1)
        class_embed_out = self.class_embedding(class_out).unsqueeze(dim=1)
        # print(f"Class embed in:{class_embed_in.shape}")
        # print(f"class embed out:{class_embed_out.shape}")

        # prepend class-embedding to encoder and decoder
        input_seq =  torch.cat([ class_embed_in,  input_seq], dim=1 )
        target_seq = torch.cat([ class_embed_out, target_seq], dim=1 )
        # print(f"Concatenated input:{ input_seq.shape}")
        # print(f"Concatenated output:{ target_seq.shape}")

        # print("END OF forward pass printing")
        # calculate encoder representations from the input: encoder output
        encoder_reps = input_seq
        for encoder_layer in self.encoder:
            encoder_reps, encoder_attention = encoder_layer( encoder_reps )

        # calculate decoder representations: use the already calculated encoder reps
        decoder_reps = target_seq
        for decoder_layer in self.decoder:
            decoder_reps, decoder_self_att, decoder_cross_att = decoder_layer( decoder_reps, encoder_reps )

        # pass the decoder representation through a linear layer
        decoder_output = self.linear_output( decoder_reps )

        # return everything
        return( decoder_output, decoder_reps, decoder_self_att, decoder_cross_att, encoder_reps, encoder_attention )




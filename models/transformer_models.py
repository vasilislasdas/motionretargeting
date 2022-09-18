import torch
import torch.nn as nn

from transfomerlayer import EncoderLayer, DecoderLayer,PositionalEncoding
import math
import sys
sys.path.append( '../utils')
# from utils.datasetRetargeting import  RetargetingDataset
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader

# pure multi-layered transformer encoder
class EncoderMultiLayer( nn.Module ):
    def __init__( self, nr_layers:int, dim_model: int, nr_heads: int = 1, dropout : float = 0.1, dim_inter: int = None ):
        super( EncoderMultiLayer, self ).__init__()

        self.nr_layers = nr_layers
        self.dim_model = dim_model
        self.layers = nn.ModuleList( [ EncoderLayer(dim_model=self.dim_model,
                                                    nr_heads=nr_heads,
                                                    dropout=dropout,
                                                    dim_inter=dim_inter ) for _ in range(self.nr_layers) ] )

    def forward( self, X ):
        reps = X
        for layer in self.layers:
            reps,attention = layer( reps )
        return reps,attention



class TransformerEncoderDecoder( nn.Module ):

    def __init__(self, nr_layers:int, dim_input:int, dim_model: int, dim_output:int = None,  nr_heads: int = 1, dropout : float = 0.1, dim_inter: int = None ):
        super(TransformerEncoderDecoder, self).__init__()

        self.nr_layers = nr_layers
        self.dim_input = dim_input # dimensionality of each symbol in the input sequence
        self.dim_model = dim_model # internal dim for both encoder decoder
        self.nr_heads = nr_heads
        self.dropout= dropout
        self.dim_inter = dim_inter
        self.dim_output = dim_output # dimensionality of each symbol in the output sequence

        # projects the input to the internal dimensionality of the model
        self.linear_input = nn.Linear( self.dim_input, self.dim_model )

        # projects the output of the transformers to the desired dimenionsnality
        if self.dim_output is None:
            self.dim_output = dim_input
        self.linear_output = nn.Linear( self.dim_model, self.dim_output )

        # positional embeddings
        self.positions = PositionalEncoding(d_model= self.dim_model)

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



    def forward( self, input_seq:torch.Tensor, target_seq:torch.Tensor):

        # make sure we are given 3D-tensors for the encoder and decoder: batch_size X seq_len X input_dim
        assert(input_seq.ndim == 3 and target_seq.ndim == 3 )

        # first project the dimensionality of the input to the internal dim of the encoder-decoder blocks
        input_seq = self.linear_input(input_seq )
        target_seq = self.linear_input(target_seq )

        # add positional embeddings
        input_seq = self.positions(input_seq)
        targe_seq = self.positions(target_seq)

        # calculate encoder representations from the input: encoder output
        encoder_reps = input_seq
        for encoder_layer in self.encoder:
            encoder_reps, encoder_attention = encoder_layer( encoder_reps )


        # calculate decoder representations: use the already calculated encoder reps
        decoder_reps = target_seq
        for decoder_layer in self.decoder:
            decoder_reps, decoder_self_att, decoder_cross_att = decoder_layer( decoder_reps, encoder_reps )

        # pass the decoder representation through a linear layer
        decoder_output = self.linear_output(decoder_reps)

        # return evrything
        # return( decoder_reps, decoder_self_att, decoder_cross_att, encoder_reps, encoder_attention )
        return( decoder_output, decoder_reps, decoder_self_att, decoder_cross_att, encoder_reps, encoder_attention )


class Retargeter( nn.Module ):

    def __init__(self, nr_layers:int, dim_input:int, dim_model: int, dim_output:int = None,
                 seq_length:int=1, nr_heads: int = 1, dropout : float = 0.1, dim_inter: int = None, skeleton_dim: int = 135 ):
        super(Retargeter, self).__init__()

        self.nr_layers = nr_layers
        self.dim_input = dim_input # dimensionality of each symbol in the input sequence
        self.dim_model = dim_model # internal dim for both encoder decoder
        self.nr_heads = nr_heads
        self.dropout= dropout
        self.dim_inter = dim_inter
        self.dim_output = dim_output # dimensionality of each symbol in the output sequence
        self.skeleton_dim = skeleton_dim
        self.seq_length = seq_length

        # projects the input to the internal dimensionality of the model
        self.linear_input = nn.Linear( self.dim_input, self.dim_model )
        # option 2 instead of a linear use a convolution
        # self.linear_input2 = torch.nn.Conv1d( in_channels=dim_input, out_channels=dim_model, kernel_size=3, padding='same')


        # projects the output of the transformers to the desired dimenionsnality
        if self.dim_output is None:
            self.dim_output = dim_input
        self.linear_output = nn.Linear( self.dim_model* self.seq_length, self.dim_output*seq_length )
        # self.linear_output2 = nn.Linear( self.dim_output, self.dim_output)
        # self.relu = nn.LeakyReLU(0.1)


        # positional embeddings
        self.positions = PositionalEncoding(d_model= self.dim_model)

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

        # embeddings for the skeleton: 25 skeletons in mixamo, when flattened 135(after padding)
        # has same dimensions as the linear projection of input in order to be able to stack it
        self.embed_skeleton = nn.Linear( self.skeleton_dim, self.dim_model )


    def forward( self, input_seq:torch.Tensor, target_seq:torch.Tensor, input_skeleton, output_skeleton ):

        # make sure we are given 3D-tensors for the encoder and decoder: batch_size X seq_len X input_dim
        assert(input_seq.ndim == 3 and target_seq.ndim == 3 )

        # first project the dimensionality of the input to the internal dim of the encoder-decoder blocks
        input_seq = self.linear_input(input_seq )
        target_seq = self.linear_input(target_seq)

        # add positional embeddings
        input_seq = self.positions(input_seq)
        target_seq = self.positions(target_seq)
        # print(f"Input seq:{input_seq.shape}")
        # print(f"Output seq:{target_seq.shape}")

        # project input and output skeleton
        input_skeleton_embedding = self.embed_skeleton( input_skeleton )
        output_skeleton_embedding = self.embed_skeleton( output_skeleton )

        # concatenate skeleton with sequence to create a new seq
        input_seq = torch.cat( [ input_skeleton_embedding, input_seq], dim=1)
        target_seq = torch.cat([ output_skeleton_embedding, target_seq], dim=1)
        # print(f"Concatenated input:{input_seq.shape}")
        # print(f"Concatenated output:{target_seq.shape}")

        # calculate encoder representations from the input: encoder output
        encoder_reps = input_seq
        for encoder_layer in self.encoder:
            encoder_reps, encoder_attention = encoder_layer( encoder_reps )


        # calculate decoder representations: use the already calculated encoder reps
        decoder_reps = target_seq
        for decoder_layer in self.decoder:
            decoder_reps, decoder_self_att, decoder_cross_att = decoder_layer( decoder_reps, encoder_reps )

        # pass the decoder representation through a linear layer but make sure to flatten
        decoder_output = self.linear_output( decoder_reps.view( decoder_reps.shape[0], -1 ) )
        # decoder_output = self.linear_output( decoder_reps )
        # decoder_output = self.linear_output2( self.relu( self.linear_output( decoder_reps ) ) )

        # return everything
        return( decoder_output, decoder_reps, decoder_self_att, decoder_cross_att, encoder_reps, encoder_attention )




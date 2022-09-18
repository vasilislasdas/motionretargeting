from builtins import int
import torch
import torchinfo
import torch.nn as nn
from transfomerlayer import EncoderLayer
from itertools import product

if __name__=="__main__":

    # run a full test on the encoder with various values
    input_dims = (2 ** torch.arange(10)).type(torch.int)
    heads= (2 ** torch.arange( 4)).type(torch.int)
    batches = 2 ** torch.arange( 5 )
    batches = (torch.Tensor([1])).type(torch.int)
    seq_len_encoder = torch.arange( 1,1024,10 ).type(torch.int)
    # inter_dim = 2
    gpu = False

    values = product( batches, input_dims, seq_len_encoder, heads)

    for batch, dim, seq_len, head in values:

        batch = batch.item()
        dim = dim.item()
        seq_len = seq_len.item()
        head = head.item()

        # create an encoder: make sure to skip invalid heads and dim combination
        if   dim % head != 0:
            continue

        # create input
        inp = torch.rand( batch, seq_len, dim )
        # print(f"Input shape:{inp.shape}")
        # print( f"batch:{batch}, dim:{dim}, seq_len:{seq_len}, head:{head}" )

        encoder = EncoderLayer( dim_model=dim, nr_heads=head )
        # move to gpu
        if gpu:
            inp = inp.to(device="cuda")
            encoder = encoder.to(device="cuda")

        encoder_reps, attention = encoder(inp)
        print( f"encoder reps:{encoder_reps.shape}, attention:{attention.shape}")



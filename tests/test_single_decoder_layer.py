import torch
import torchinfo
import torch.nn as nn
from itertools import product
import sys
sys.path.append( '../models')
from transfomerlayer import EncoderLayer, DecoderLayer


if __name__=="__main__":

    # run a full test on the encoder with various values
    input_dims = (2 ** torch.arange(10)).type(torch.int)
    heads = (2 ** torch.arange(4)).type(torch.int)
    batches = 2 ** torch.arange(5)
    # batches = (torch.Tensor([11])).type(torch.int)
    seq_len_encoder = torch.arange(1, 1024, 50).type(torch.int)
    seq_len_decoder = torch.arange(1, 1024, 50).type(torch.int)
    # inter_dim = 2
    gpu = False

    values = product(batches, input_dims, seq_len_encoder, heads, seq_len_decoder )

    for batch, dim, seq_len, head, seq_len2 in values:

        batch = batch.item()
        dim = dim.item()
        seq_len = seq_len.item()
        head = head.item()

        # create an encoder: make sure to skip invalid heads and dim combination
        if dim % head != 0:
            continue

        # create input
        inp = torch.rand(batch, seq_len, dim)
        # print(f"Input shape:{inp.shape}")
        print( f"batch:{batch}, dim:{dim}, seq_len_encoder:{seq_len}, head:{head}, seq_len_decoder:{seq_len2}" )

        encoder = EncoderLayer(dim_model=dim, nr_heads=head)

        encoder_reps, encoder_attention = encoder(inp)
        print(f"encoder reps:{encoder_reps.shape}, attention:{encoder_attention.shape}")

        # test the decoder layer
        decoder = DecoderLayer( dim_model=dim, nr_heads=head )

        # move to gpu if desired
        if gpu:
            inp = inp.to(device="cuda")
            encoder = encoder.to(device="cuda")
            decoder = decoder.to(device="cuda")

        target = torch.rand(batch, seq_len2, dim)
        decoder_reps, self_attention, cross_attention = decoder(target, encoder_reps)
        print("decoder reps:", decoder_reps.shape, self_attention.shape, cross_attention.shape, '\n')


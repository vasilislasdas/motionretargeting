import torch
import torchinfo
import torch.nn as nn

if __name__="__main__":

    # some input
    input_dim = 512
    inter_dim = 512
    hidden_dim = 512
    batch = 1
    seq_len = 60
    seq_len_decoder = 59
    heads = 8
    inp = torch.rand(batch, seq_len, input_dim)
    target = torch.rand(batch, seq_len_decoder, input_dim)
    gpu = False
    if gpu:
        inp = inp.to(device="cuda")  # move to gpu
        target = target.to(device="cuda")  # move to gpu

    # test the encoder layer
    encoder = EncoderLayer(dim_input=input_dim, dim_hidden=input_dim, nr_heads=heads)
    if gpu:
        encoder = encoder.to(device="cuda")  # move to gpu
    encoder_reps, attention = encoder(inp)
    # torchinfo.summary(encoder, input_data=inp)
    print("encoder reps:", encoder_reps.shape, attention.shape)
    # print(encoder)
    # for name, param in encoder.named_parameters():
    #     print("Parameter %s, shape %s" % (name, str(param.shape)))

    # test the decoder layer
    decoder = DecoderLayer(dim_target=input_dim, dim_hidden=hidden_dim, nr_heads=heads, dim_inter=inter_dim)
    if gpu:
        decoder = decoder.to(device="cuda")

    decoder_reps, self_attention, cross_attention = decoder(target, encoder_reps)
    print("decoder reps:", decoder_reps.shape, self_attention.shape, cross_attention.shape)
    # torchinfo.summary(decoder, input_data=(inp, encoder_reps))

    # test a complete encoder composed only from encoder-layers
    # network = EncoderMultiLayer( nr_layers=4, dim_input=input_dim, dim_hidden=hidden_dim,nr_heads=8 )
    # start = time.time()
    # out,attention = network(inp )
    # print(f"Total time:{time.time()-start}")
    # torchinfo.summary(network, input_data=(inp))
    # print(f"Out:{out.shape}")
    # print(f"Attention:{attention.shape}")

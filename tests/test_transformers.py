import torch
import torch.utils.data as data
import torch
import torch.nn as nn
import sys
sys.path.append( '../models')
sys.path.append( '../utils')
from transformer_models import TransformerEncoderDecoder
from datasetReverse import ReverseSequences, ReverseSequenceSummed
import torchinfo
import tqdm


# global variables
seq_len = 30
nr_layers = 3
nr_classes = 10 # i predict numbers between 0-9(10 excluded) # when using classification loss
input_dim = 11
output_dim= 11
model_dim = 128
heads=4
training_size = 50000
epochs = 30

device='cuda'
batch_training_size = 100
crossloss = False




def train_model( model, optimizer, data_loader, loss_module, device = 'cpu', num_epochs=1):

    ###### Set model to train mode
    model.train()
    model.to(device)
    print("STARTING TRAINING")

    # Training loop
    # for epoch in tqdm.tqdm(range(num_epochs)):
    for epoch in (range(num_epochs)):

        train_iterator = iter(data_loader)
        nb_batches_train = len(data_loader)
        train_acc = 0
        model.train()
        losses = 0.0

        # for data_inputs, data_labels in data_loader:
        # for batch in train_iterator:
        for data_inputs, data_labels in train_iterator:
            optimizer.zero_grad()

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            inp, target = data_inputs.to(device), data_labels.to(device)
            # print( "Input,target:" , input.shape, target.shape )

            # Step 2: Run the model on the input data
            target_seq = target.detach().clone()
            out_seq = target.detach().clone()
            target_seq = target_seq[:, :-1, :]
            out_seq = out_seq[:, 1:, :]

        #     # unsqueeze input and target
        #     input_seq = input.unsqueeze( dim = -1 )
        #     target_seq = target_seq.unsqueeze(dim=-1)
        #     out_seq = out_seq.unsqueeze(dim=-1)
            # print("Input,target:", input.shape, target_seq.shape)
            # print(f'Encoder input seq:{input_seq}')
            # print( f'Decoder input seq:{target_seq}, "\n" Decoder out seq:{out_seq}' )
            preds = model( inp, target_seq )


            # Step 3: Calculate the loss
            tmp = preds[0]
            # print("preds:", tmp)
            # print("out:", out_seq)
            if crossloss is False:
                loss = loss_module(tmp, out_seq )
            else:
                out_seq = torch.reshape(out_seq, (-1,))
                tmp = torch.reshape(tmp, ( tmp.shape[0]*tmp.shape[1],-1 ) )
                loss = loss_module(tmp, out_seq.long())



            # Perform backpropagation
            loss.backward()
            losses += loss.item()

            ## Step 5: Update the parameters
            optimizer.step()

        print(f"Training loss at epoch {epoch} is {losses / nb_batches_train}")

    print("TRAINING FINISHED")


def test_transformer_creation( show_fullinfo = False):
    # some input
    input_dim = 1
    model_dim = 16
    inter_dim = 16
    batch = 3
    seq_len = 11
    seq_len_decoder = 5
    heads = 1
    nr_layers = 1
    dropout = 0
    inp = torch.rand(batch, seq_len, input_dim)
    target = torch.rand(batch, seq_len_decoder, input_dim)

    print( f"Input dim:{input_dim}, model_dim:{model_dim}, encoder seq_len:{seq_len},"
           f" decoder_seq_len:{seq_len_decoder}, heads:{heads}, batch:{batch}")

    gpu = False
    if gpu:
        inp = inp.to(device="cuda")  # move to gpu
        target = target.to(device="cuda")  # move to gpu

    transformer = TransformerEncoderDecoder(nr_layers=nr_layers,
                                            dim_input= input_dim,
                                            dim_model=model_dim,
                                            nr_heads=heads,
                                            dropout=dropout,
                                            dim_inter=inter_dim)
    # transformer.eval()
    out = transformer( inp, target)
    # for el in out:
    #     print(el.shape)

    # print details info:
    print( f"Transformer linear layer decoder output:{out[0].shape}")
    print( f"Transformer decoder representation:{out[1].shape}")
    print(f"Transformer decoder self-attention:{out[2].shape}")
    print(f"Transformer decoder cross-attention:{out[3].shape}")
    print( f"Transformer encoder representation:{out[4].shape}")
    print( f"Transformer encoder self-attention:{out[5].shape}")

    if show_fullinfo:
        torchinfo.summary(transformer, input_data=(inp, target))

    return transformer

def eval_reverse_seq():

    # create model
    transformer = TransformerEncoderDecoder(nr_layers=nr_layers,
                                            dim_input=input_dim,
                                            dim_model=model_dim,
                                            dim_output=output_dim,
                                            nr_heads=heads,
                                            dropout=0.1)

    # restore model
    state_dict = torch.load("transformer.zip")
    transformer.load_state_dict(state_dict)

    # evaluate model: create a new random dataset
    transformer.eval()
    # eval_dataset = ReverseSequences(seq_len=seq_len, dataset_size=1)
    eval_dataset = ReverseSequenceSummed(seq_len=seq_len, dataset_size=training_size,dims=input_dim)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    inp_seq,target_seq = next(iter(eval_loader))
    first_el = inp_seq[ :, -1, : ] # start with the first symbold
    start_seq = first_el[None, :, :]

    # print(f"Test input sequence:{input_seq}, start_seq{start_seq}")
    if crossloss == False:
        for i in range(seq_len - 1):
            decoder_out = transformer(inp_seq, start_seq)[0] # get transformer output
            last_pred = decoder_out[:, -1, :].unsqueeze(0) # pick last prediction coz this is the one that we are interested
            start_seq = torch.cat((start_seq, last_pred), dim=1)

            # print(f"concatenated_start_seq:{start_seq}")
        # expected =  torch.flip(inp_seq, dims=[1]).flatten()
        # print(f"Expec:{expected.int()}")
        print(f"Groundthruth:{target_seq}")
        print(f"Pred :{torch.round(start_seq)}")
    else:
        for i in range(seq_len - 1):
            decoder_out = transformer( inp_seq, start_seq )[0] # decoder output
            logits = decoder_out[:, -1, :].flatten()
            probs=nn.functional.softmax(logits, dim = 0)
            number= torch.argmax(probs).item()
            last_pred = torch.Tensor([number])[None,:,None]
            start_seq = torch.cat((start_seq, last_pred), dim=1)


        expected = torch.flip(input_seq, dims=[1]).flatten()
        print(f"Input:{input_seq.flatten()}")
        print(f"Prediction:{start_seq.flatten()}")
        print(f"Expected:{expected.flatten()}")




def train_reverse_seq():
    # dataset

    # dataset = ReverseSequences(seq_len=seq_len, dataset_size=training_size)
    dataset = ReverseSequenceSummed(seq_len=seq_len, dataset_size=training_size,dims=input_dim)
    data_loader = data.DataLoader(dataset, batch_size=batch_training_size, shuffle=False)

    # create model
    transformer = TransformerEncoderDecoder(nr_layers=nr_layers,
                                            dim_input=input_dim,
                                            dim_model=model_dim,
                                            dim_output= output_dim,
                                            nr_heads=heads,
                                            dropout=0.1)
    # loss function and optimizer
    if crossloss == True:
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.MSELoss()


    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(transformer.parameters(), lr=0.1)

    # perform train
    train_model( model=transformer, optimizer=optimizer, data_loader=data_loader, loss_module = loss, device = device, num_epochs=epochs )
    state_dict = transformer.state_dict()
    torch.save( state_dict, "transformer.zip")



if __name__ == "__main__":
    pass
    # test_transformer_creation()
    # train_reverse_seq()
    eval_reverse_seq()


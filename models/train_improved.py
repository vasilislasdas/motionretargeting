import torch
import torch.utils.data as data
import torch
import torch.nn as nn
import sys
sys.path.append( '../models')
sys.path.append( '../utils')
from retargeterGenerator import RetargeterGenerator
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader
from shutil import copyfile, copy2
from bvh_manipulation import *

##### specify model hyperparameters
nr_layers = 8
heads = 5
model_dim = 100
device = 'cuda'
epochs = 1
batch_size = 200
shuffle_data = True
print(f"Layers:{nr_layers}")
print(f"Number of heads used:{heads}")
print(f"Model dimension:{model_dim}")
print(f"Epochs:{epochs}")
print(f"Batch size:{batch_size}")
print(f"Shuffle data:{shuffle_data}")



def train_mixamo():

    print("PREPARING TRAINING....")

    ###### create:dataset and dataloader fpr mixamo dataset
    dataset = RetargetingDataset( dataset_file="../utils/normalized_dataset.txt" )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=1)
    print(f"Total number of batches:{ len(data_loader) }" )


    # fetch first sample to get dimensions of the motion and the skeleton
    first_sample = dataset[0]
    motion = first_sample[0]
    flat_skeleton = first_sample[3]
    seq_len, input_dim = motion.shape
    skeleton_dim = flat_skeleton.shape[1]

    output_dim = input_dim
    input_dim = input_dim + skeleton_dim
    print(f"input_dim:{input_dim}")


    ##### create retargeting model
    retargeter = RetargeterGenerator( nr_layers=nr_layers,
                              dim_input=input_dim,
                              dim_model=model_dim,
                              dim_output= output_dim,
                              nr_heads=heads,
                              dropout=0.1 )


    #### loss function
    reconstruction_loss = nn.MSELoss()
    time_loss = nn.MSELoss()


    ##### perform training
    train_model(model=retargeter, data_loader=data_loader, rec_loss_module=reconstruction_loss,
                time_loss_module=time_loss, device=device, num_epochs=epochs)

    ##### save the created network
    state_dict = retargeter.state_dict()
    torch.save( state_dict, "retargeter.zip")


def train_model( model, data_loader, rec_loss_module, time_loss_module, device = 'cpu', num_epochs=1):
    print("STARTING TRAINING")


    # Set model to train mode and move it to the desired device
    model.train()
    model.to( device )

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ####### Training loop
    for epoch in (range(num_epochs)):

        print(f"Epoch:{epoch}")

        # create iterator from dataloader
        train_iterator = iter(data_loader)

        # get batch size
        nb_batches_train = len(data_loader)

        # keep track of the loss over each epoch
        self_losses_positions = 0.0
        self_losses_rotations = 0.0
        time_losses = 0.0
        total_losses = 0.0

        ##### training loop over batches
        for batch in train_iterator:

            # clean gradient
            optimizer.zero_grad()

            # unpack info from input batch
            # (motion, character, type, flat_skeleton.float(), maximum, minimum)
            motion = batch[0]
            character_name = batch[1]
            skeleton = batch[3]
            # print(character_name)

            # Move input data to device for motion, character, sequence
            input_seq = motion.to(device)
            input_skeleton = skeleton.to(device)
            classes_in = create_one_hot_from_name(character_name)
            classes_in = classes_in.to(device)


            # create target sequence, output skeleton, output char name: for self-loss just copy-past and be careful for the target and expected_sequence
            target_seq = input_seq.detach().clone()
            target_seq = target_seq[:, :-1, :]
            expected_seq = input_seq.detach().clone() # this is what we want the decoder to generate
            output_character_name = character_name
            classes_out = create_one_hot_from_name(output_character_name)
            classes_out = classes_in.to(device)
            output_skeleton = input_skeleton.detach().clone()


            # predict
            prediction,_,_,_,_,_ = model( input_seq, target_seq, input_skeleton, output_skeleton, classes_in, classes_out )

            #  Calculate the self-reconstruction loss
            # print( f"Prediction shape:{prediction.shape}")
            # print(f"expected_seq shape:{expected_seq.shape}")
            positions = prediction[:, 0:3, :]
            rotations = prediction[:, 3:, :]
            positions_original = expected_seq[:, 0:3, :]
            rotations_original = expected_seq[:,3:,:]

            self_loss_positions = rec_loss_module( positions_original, positions )
            self_loss_rotations = rec_loss_module( rotations_original, rotations)


            # compute time-consistency loss
            tmp1 = expected_seq[:, 1:, :] - expected_seq[:, :-1, :]
            tmp2 = prediction[:, 1:, :] - prediction[:, :-1, :]
            time_loss = time_loss_module( tmp1.view(-1), tmp2.view(-1) )

            # add losses
            total_loss = time_loss + self_loss_positions + self_loss_rotations
            # total_loss = self_loss


            # Perform backpropagation
            total_loss.backward()


            # store losses for tracking progress
            self_losses_positions += self_loss_positions.item()
            self_losses_rotations += self_loss_rotations.item()
            time_losses += time_loss.item()
            total_losses += total_loss.item()

            # total_losses += total_loss.item()

            ## Step 5: Update the parameters
            optimizer.step()

        print(f"Training self-reconstruction loss positions  at epoch {epoch} is {self_losses_positions / nb_batches_train}")
        print(f"Training self-reconstruction loss  rotations at epoch {epoch} is {self_losses_rotations / nb_batches_train}")
        print(f"Training time loss  at epoch {epoch} is {time_losses / nb_batches_train}")
        print(f"Training total loss  at epoch {epoch} is {total_losses / nb_batches_train}")



    print("TRAINING FINISHED")


def evaluate_mixamo():
    print("PREPARING EVALUATION....")

    ###### create:dataset and dataloader fpr mixamo dataset
    dataset = RetargetingDataset(dataset_file="../utils/normalized_dataset.txt")


    # fetch first sample to get dimensions of the motion and the skeleton
    first_sample = dataset[0]
    motion = first_sample[0]
    flat_skeleton = first_sample[3]
    seq_len, input_dim = motion.shape
    skeleton_dim = flat_skeleton.shape[1]
    output_dim = input_dim
    input_dim = input_dim + skeleton_dim
    minumum = motion[-1]
    maximum = motion[-2]


    ##### create retargeting model
    retargeter = RetargeterGenerator( nr_layers=nr_layers,
                              dim_input=input_dim,
                              dim_model=model_dim,
                              dim_output= output_dim,
                              nr_heads=heads,
                              dropout=0.1 )



    # restore model
    state_dict = torch.load("retargeter.zip")
    retargeter.load_state_dict(state_dict)

    # set network in evaluation mode
    retargeter.eval()


    # get a random index
    range_words = len(dataset)
    ind = torch.randint(low=0, high=range_words, size=(1,)).item()


    # pick random clip
    motion, character, motion_type, skeleton, maximum, minimum = dataset[ ind ]
    skeleton = skeleton.unsqueeze( dim=0 )

    # prepare input output
    input_seq = motion.unsqueeze(dim = 0)
    input_skeleton = skeleton
    output_skeleton = skeleton
    target_seq = input_seq[:, 0:1, : ]
    print(f"Input seq:{ input_seq.shape}" )
    print(f"Target seq:{target_seq.shape}")
    classes_in = create_one_hot_from_name([character])
    classes_out = create_one_hot_from_name([character])


    # predict in a for loop
    for i in range( seq_len - 1):

        # predict
        prediction, _, _, _, _, _ = retargeter( input_seq, target_seq,
                                                input_skeleton, output_skeleton,
                                                classes_in, classes_out )
        print(f"Prediction shape:{prediction.shape}")

        #  pick last prediction coz this is the one that we are interested
        last_pred = prediction[:, -1:, :]
        # print(f"Last prediction shape:{last_pred.shape}")

        # concatenate last prediction
        target_seq = torch.cat((target_seq, last_pred), dim=1)


    loss = nn.MSELoss()
    rec_loss = loss( input_seq, target_seq )
    print(f"Prediction loss:{rec_loss.item()}")
    print( f"Input:{input_seq[0, :5,0]}" )
    print(f"Target:{target_seq[0, :5, 0]}")
    print(f"Prediction:{prediction[0, :5, 0]}")


    # visualize prediction + original motion
    visualize_result( motion_word=motion, character=character, motion_type=motion_type, minimum=minimum, maximum=maximum, suffix = "_original.bvh" )
    visualize_result(motion_word=target_seq.squeeze(dim=0), character=character, motion_type=motion_type, minimum=minimum, maximum=maximum,
                     suffix="_predicted.bvh")



def visualize_result( motion_word, character, motion_type, minimum, maximum, suffix = ".bvh" ):

    ## copy original file from motion word locally from the mixamo folder for easier manipulation
    new_name = copy_file_locally( character, motion_type )

    # read the motion from the locally copied original file to get original edges and joint names
    original_motion, original_edges, original_names = read_motion_quaternion( new_name, downSample=False)


    # reconstruct motion
    delta = maximum - minimum
    restored_motion = ( ( motion_word + 1 ) / 2 ) * delta + minimum
    restored_motion[torch.isnan(restored_motion)] = 0
    restored_motion[torch.isinf(restored_motion)] = 0
    isNan = torch.any(restored_motion.isnan())
    isInf = torch.any(torch.isinf(restored_motion))
    assert (isNan == False and isInf == False)

    # now write the restored motion
    reconstructed_file = new_name.replace(".bvh", suffix)
    write_motion_quaternions_to_bvh_directly(motion=restored_motion, edges=original_edges, joint_names=original_names,
                                             bvh_output_file=reconstructed_file)





if __name__ == "__main__":

    # train_mixamo()
    evaluate_mixamo()





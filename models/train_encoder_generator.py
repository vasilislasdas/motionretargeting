import torch
import torch.utils.data as data
import torch
import torch.nn as nn
import sys
sys.path.append( '../models')
sys.path.append( '../utils')
from retargeterGeneratorEncoderOnly import RetargeterGeneratorEncoder
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader
from shutil import copyfile, copy2
from bvh_manipulation import *

##### specify model hyperparameters
nr_layers = 4
heads = 8
model_dim = 128
device = 'cuda'
epochs = 120
batch_size = 120
shuffle_data = True
print(f"Layers:{nr_layers}")
print(f"Number of heads used:{heads}")
print(f"Model dimension:{model_dim}")
print(f"Epochs:{epochs}")
print(f"Batch size:{batch_size}")
print(f"Device:{device}")
print(f"Shuffle data:{shuffle_data}")

def compute_self_loss( model, loss_module, input_seq, input_skeleton ):

    # create the target and output sequence for the decoder: 0...dim-1, 1...dim
    expected_seq = input_seq.detach().clone()
    target_skeleton = input_skeleton.detach().clone()

    # predict
    prediction, _, _ = model( input_seq, input_skeleton, target_skeleton)
    # print( f"Predictions:{prediction.shape}")
    # print( f"expected_seq:{expected_seq.shape}")

    # fetch positions and rotaions and calculate seperate losses
    positions = prediction[:, 0:3, :]
    rotations = prediction[:, 3:, :]
    positions_original = expected_seq[:, 0:3, :]
    rotations_original = expected_seq[:, 3:, :]

    # self-loss
    # self_loss = rec_loss_module( prediction, expected_seq )
    self_loss_positions = loss_module(positions_original, positions)
    self_loss_rotations = loss_module(rotations_original, rotations)

    return prediction, self_loss_positions, self_loss_rotations

def compute_cycle_loss(  model, loss_module, input_seq, input_skeleton, input_characters ):

    # random indices to permute the batch
    new_indices = torch.randperm( input_seq.shape[0] )

    # create random skeletons to retarget to
    target_skeleton = input_skeleton[ new_indices ]

    # give a new target skeleton and get prediction
    retargeted_motion, _, _ = model( input_seq, input_skeleton, target_skeleton )

    # feed back the prediction in the model with the skeleton and now target skeleton the original skeleton
    prediction, _, _ = model( retargeted_motion, target_skeleton, input_skeleton )

    #
    positions = prediction[:, 0:3, :]
    rotations = prediction[:, 3:, :]
    positions_original = input_seq[:, 0:3, :]
    rotations_original = input_seq[:, 3:, :]

    # self-loss
    # self_loss = rec_loss_module( prediction, expected_seq )
    cycle_loss_positions = loss_module( positions_original, positions )
    cycle_loss_rotations = loss_module( rotations_original, rotations )


    return prediction, cycle_loss_positions, cycle_loss_rotations



def compute_time_loss( time_loss_module, prediction, expected_seq ):

    # compute time-consistency loss
    predicted_diff = prediction[:, 1:, :] - prediction[:, :-1, :]
    expected_diff = expected_seq[:, 1:, :] - expected_seq[:, :-1, :]

    # time loss
    time_loss = time_loss_module(predicted_diff, expected_diff)

    return time_loss




def train_mixamo():

    print("PREPARING TRAINING....")

    ###### create:dataset and dataloader fpr mixamo dataset
    dataset = RetargetingDataset( dataset_file="../utils/normalized_dataset.txt" )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=1)
    print(f"Total number of batches:{ len(data_loader) }" )


    # fetch first sample to get dimensions of the motion and the skeleton
    first_sample = dataset[0]
    motion = first_sample[0]
    skeleton = first_sample[3]
    seq_len, input_dim = motion.shape
    output_dim = input_dim

    ##### create retargeting model
    retargeter = RetargeterGeneratorEncoder(nr_layers=nr_layers,
                            dim_input=input_dim,
                            dim_model=model_dim,
                            seq_len=seq_len,
                            dim_skeleton= skeleton.shape[1],
                            nr_heads=heads,
                            dropout=0.1)

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

        # keep track of the losses over each epoch

        # self-related losses
        self_losses_positions = 0.0
        self_losses_rotations = 0.0
        time_losses_self = 0.0

        # cycle-related losses
        cycle_losses_positions = 0.0
        cycle_losses_rotations = 0.0
        time_losses_cycle = 0.0

        # total loss
        total_losses = 0.0

        ##### training loop over batches
        for batch in train_iterator:

            # clean gradient
            optimizer.zero_grad()

            # unpack info from input batch
            motion = batch[0]
            skeleton = batch[3]
            character = batch[1]
            # print( f"Motion length:{motion.shape}" ), print(f"Flat skeleton:{skeleton.shape}")

            # Move input data to device
            input_seq = motion.to( device)
            input_skeleton = skeleton.to( device )


            # compute self-reconstruction loss
            prediction, self_loss_positions, self_loss_rotations = compute_self_loss( model=model,
                                                                                      loss_module=rec_loss_module,
                                                                                      input_seq=input_seq,
                                                                                      input_skeleton=input_skeleton)


            # compute time-consistency self-reconstruction loss
            time_loss_self = compute_time_loss( time_loss_module=time_loss_module,
                                                prediction = prediction,
                                                expected_seq=input_seq.detach().clone() )


            prediction_cycle, cycle_loss_positions, cycle_loss_rotations = \
                compute_cycle_loss( model=model,
                                    loss_module=rec_loss_module,
                                    input_seq=input_seq,
                                    input_skeleton=input_skeleton,
                                    input_characters=character)


            # compute time-consistency cycle-reconstruction loss
            time_loss_cycle = compute_time_loss( time_loss_module=time_loss_module,
                                                 prediction=prediction_cycle,
                                                 expected_seq=input_seq.detach().clone())


            # add losses
            total_loss = self_loss_positions +  self_loss_rotations + time_loss_self + \
                         cycle_loss_positions + cycle_loss_rotations + time_loss_cycle


            # Perform backpropagation
            total_loss.backward()

            # store losses for tracking progress
            self_losses_positions += self_loss_positions.item()
            self_losses_rotations += self_loss_rotations.item()
            time_losses_self += time_loss_self.item()
            cycle_losses_positions += cycle_loss_positions.item()
            cycle_losses_rotations += cycle_loss_rotations.item()
            time_losses_cycle += time_loss_cycle.item()
            total_losses += total_loss.item()

            # total_losses += total_loss.item()

            ## Step 5: Update the parameters
            optimizer.step()

        # print( f"Training self-reconstruction loss at epoch {epoch} is {self_losses / nb_batches_train}")
        print( f"Training self-reconstruction loss positions at epoch {epoch} is {self_losses_positions / nb_batches_train}")
        print( f"Training self-reconstruction loss rotations at epoch {epoch} is {self_losses_rotations / nb_batches_train}")
        print( f"Training time loss self  at epoch {epoch} is {time_losses_self / nb_batches_train}")
        print( f"Training cycle-reconstruction loss positions at epoch {epoch} is {cycle_losses_positions / nb_batches_train}")
        print( f"Training cycle-reconstruction loss rotations at epoch {epoch} is {cycle_losses_rotations / nb_batches_train}")
        print( f"Training time loss cycle  at epoch {epoch} is {time_losses_cycle / nb_batches_train}")
        print(f"Training total loss  at epoch {epoch} is {total_losses / nb_batches_train}")



    print("TRAINING FINISHED")


def evaluate_mixamo():
    print("PREPARING EVALUATION....")

    ###### create:dataset and dataloader fpr mixamo dataset
    dataset = RetargetingDataset(dataset_file="../utils/normalized_dataset.txt")

    # get a random index
    range_words = len(dataset)
    ind = torch.randint(low=0, high=range_words, size=(1,)).item()
    target_ind = torch.randint(low=0, high=range_words, size=(1,)).item()


    # pick random clip
    motion, character, motion_type, skeleton, maximum, minimum = dataset[ind]
    seq_len, input_dim = motion.shape


    # prepare input output
    input_seq = motion.unsqueeze(dim=0)
    skeleton = skeleton.unsqueeze(dim=0)
    input_skeleton = skeleton


    ##### create retargeting model
    retargeter = RetargeterGeneratorEncoder(nr_layers=nr_layers,
                            dim_input=input_dim,
                            dim_model=model_dim,
                            seq_len=seq_len,
                            dim_skeleton= skeleton.shape[-1],
                            nr_heads=heads,
                            dropout=0.1)

    # restore model
    state_dict = torch.load("retargeter.zip")
    retargeter.load_state_dict(state_dict)

    # random character
    motion_target, character_target, motion_type_target, skeleton_target, maximum, minimum = dataset[target_ind]
    seq_len, input_dim = motion.shape
    skeleton_target = skeleton_target.unsqueeze(dim=0)






    # set network in evaluation mode
    retargeter.eval()


    # predict output seq from input seq: should be the smae
    prediction,_,_ = retargeter( input_seq=input_seq, input_skeleton =input_skeleton, target_skeleton=skeleton_target)

    # show some stuff
    loss = nn.MSELoss()
    rec_loss = loss( input_seq, prediction )
    print(f"Prediction loss:{rec_loss.item()}")
    print( f"Input:{input_seq[0, :5,0]}" )
    print(f"Prediction:{prediction[0, :5, 0]}")


    # visualize prediction + original motion
    visualize_result( motion_word=motion, character=character, motion_type=motion_type, minimum=minimum, maximum=maximum, suffix = "_original.bvh" )
    visualize_result( motion_word=prediction.squeeze(dim=0), character=character_target, motion_type=motion_type_target, minimum=minimum, maximum=maximum,
                     suffix="_predicted.bvh")




def visualize_result( motion_word, character, motion_type, minimum, maximum, suffix = ".bvh" ):

    ## copy original file from motion word locally from the mixamo folder for easier manipulation
    new_name = copy_file_locally( character, motion_type )

    # read the motion from the locally copied original file to get original edges and joint names
    original_motion, original_edges, original_names = read_motion_quaternion( new_name, downSample=True)


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





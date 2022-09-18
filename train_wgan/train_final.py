import os

import einops
import torch.utils.data as data
import torch
import torch.nn as nn
import sys

path = os.path.dirname(os.getcwd())
# print( path )
sys.path.append( path )
sys.path.append( path + '/models')
sys.path.append( path + '/utils')
# print(sys.path)



from torch.utils.data import DataLoader
from shutil import copyfile, copy2
from preprocess import *

from configuration import fetch_conf_parameters
import time
from auxilliary_training import *
from datasetRetargeting import *
# TODO: this needs to be disabled during final training to speed up time
torch.autograd.set_detect_anomaly(False)
from discriminator import Discriminator
import itertools
# from retargeterGeneratorEncoder_latent import RetargeterGeneratorEncoderLatent
# from retargeterGeneratorEncoder import RetargeterGeneratorEncoder
from retargeterGeneratorEncoderBoth import RetargeterGeneratorEncoderBoth
import string
import random
import numpy as np

# speed up things
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# how often to run the discriminator + the generator adversarial losses
mod_epochs_disc = 1
mod_epochs_gen = 4

# how often to validate the model w.r.t. to the pretrained ones
mod_epochs_valid = 10

ee_distances = torch.Tensor([[105.4091, 105.5677,  60.4370, 110.5378, 110.5337],
        [106.7205, 106.9004,  58.8196, 111.9990, 112.0167],
        [109.8536, 110.3405,  34.3389,  80.8473,  80.8467],
        [ 94.2672,  94.2674,  49.1569,  91.1340,  91.1909],
        [ 93.8005,  93.8085,  81.7985, 153.2321, 153.2390],
        [119.1226, 119.1227,  83.6445, 171.4890, 171.4890],
        [109.2159, 109.2083,  60.3685, 113.6815, 113.6898],
        [114.9367, 114.9366,  60.2768, 141.4039, 141.4039],
        [ 94.0746,  94.0679,  48.4980,  95.9978,  96.0245],
        [130.6156, 130.7327,  66.8159, 122.3849, 122.3979],
        [131.9940, 131.9940,  67.9091, 126.8546, 126.8418],
        [ 97.6337,  97.4012,  58.9228, 114.0205, 113.6286],
        [112.0439, 111.5082,  56.0104, 110.9046, 110.9033],
        [113.3504, 113.3503,  53.0202,  97.9383,  97.8843],
        [108.0040, 108.3312,  64.1815, 120.5920, 120.5872],
        [ 75.3084,  74.9250,  41.1732,  82.2855,  82.2858],
        [ 84.6853,  84.7509,  68.9837, 125.8265, 126.0359],
        [226.5735, 226.6246, 126.0995, 229.1658, 227.7663],
        [219.8436, 219.8117, 123.0113, 227.0261, 227.0232],
        [ 93.7100,  93.7097,  54.6462,  92.3894,  92.3894],
        [ 92.8754,  92.8754,  59.1531,  99.9202,  99.9202],
        [124.3947, 124.3948,  76.4333, 138.3690, 138.3690],
        [125.2390, 125.2622,  71.8422, 134.0782, 134.1483],
        [110.0688, 110.2237,  55.4319, 116.7279, 116.7226],
        [230.1678, 229.4203, 114.9481, 221.1271, 221.1236]])



def train_network( generator, discriminator, data_loader, device = 'cpu', num_epochs=1, lr_gen = 0.001, lr_dis = 0.001,
                   extra_data = None, validator = None, mu_sigma = None, h_orig = None ):


    print("WARMING UP TRAINING")
    assert( extra_data is not None )
    assert (validator is not None)
    assert (mu_sigma is not None)
    assert (h_orig is not None)

    ##### show warning if gpu is not used
    if device != "cuda":
        print("WARNING!!!!: using cpu")

    #### lists used to record losses over a trial
    stored_gen_seperate = []
    stored_gen = []
    stored_dis = []


    ##### Set generator and discriminator to train mode and move them to device
    generator.train()
    generator.to( device )
    discriminator.train()
    discriminator.to( device )


    ## set validator to eval mode and move to device
    validator.to(device)
    validator.eval()



    ##### create optimizers for generator and discriminator
    optimizer_gen = torch.optim.Adam( generator.parameters(), lr_gen )
    # optimizer_dis = torch.optim.Adam( discriminator.parameters(), lr_dis )
    optimizer_dis = torch.optim.RMSprop(discriminator.parameters(), lr_dis)


    ### create losses for the generator
    self_loss_modules = get_self_module_losses_generator()
    cycle_loss_modules = get_cycle_module_losses_generator()
    # gen_disc_loss_module = nn.CrossEntropyLoss()
    # gen_disc_loss_module = nn.MSELoss()


    #### discriminator loss
    # disc_loss_module = nn.CrossEntropyLoss()
    # disc_loss_module = nn.MSELoss()

    # do not train discriminator in every epoch
    wass_iter = 1

    # when to store the model
    previous_fid = 1000
    previous_best = "nothing"

    # fetch mu+sigma
    mu = mu_sigma["mu_real"]
    sigma = mu_sigma["sigma_real"]


    ####### Training loop
    print("STARTING TRAINING")

    # for epoch in (range(511,num_epochs)):
    for epoch in (range(num_epochs)):

        print(f"Epoch:{epoch}")

        # get batch size
        nb_batches_train = len(data_loader)
        print(f"Batches:{nb_batches_train}")

        # keep track of the losses over each epoch
        generator_losses = 0.0
        seperate_losses_self = [0.0] * 12

        # discriminator losses
        discriminator_losses = 0.0

        # show elapsed time
        start_time = time.time()

        ##### training loop over batches
        # for index, batch in enumerate( data_loader ):
        for batch in data_loader:

            # print(f"{batch_nr}:NEW BATCH!!")
            # batch_nr +=1
            # print(f"Index:{batch[-1]}")

            ## unpack info from input batchs
            motions = batch[1]
            flat_skeletons = batch[3]

            ##  concatenate motions and skeletons and move to device
            input_seq = torch.cat( motions, dim=0 ).to( device )
            input_skeleton = torch.cat( flat_skeletons, dim=0).to( device )
            # print(f"All motions from batch:{ input_seq.shape}, device: {input_seq.device}")
            # print(f"All skeletons from batch:{input_skeleton.shape}, device: {input_skeleton.device}")
            characters = batch[ 0 ]


            # # ##### train discriminator not in every epoch to give time to the generator to adapt
            if (epoch%mod_epochs_disc) == 0:

                for i in range(wass_iter):
                    # print("training discriminator")
                    optimizer_dis.zero_grad()

                    # Calculate discriminator loss
                    skip = 1
                    discriminator_loss = get_disc_loss(generator=generator, discriminator=discriminator,
                                                       input_seq=input_seq[::skip], input_skeleton=input_skeleton[::skip] )

                    # Update gradients
                    discriminator_loss.backward()

                    # Update optimizer
                    optimizer_dis.step()

                    # keep track of generator loss
                    discriminator_losses += discriminator_loss.item()

                    # Weight clipping
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)


            ######### train generator
            optimizer_gen.zero_grad()

            all_gen_losses = get_generator_loss( generator=generator, discriminator=discriminator,
                                                 input_seq= input_seq, input_skeleton=input_skeleton,
                                                 self_loss_modules=self_loss_modules,
                                                 cycle_loss_modules=cycle_loss_modules,
                                                 epoch=epoch, extra_data=extra_data, characters=characters, h_orig = h_orig )

            # sum all the losses
            generator_loss = sum( all_gen_losses )
            generator_loss.backward()
            optimizer_gen.step()

            ## keep track of generator loss
            generator_losses += generator_loss.item()
            for index,loss in enumerate(all_gen_losses):
                seperate_losses_self[ index ] += loss.item()

        #### print losses and save them
        gen_epoch_losses = []
        for item in seperate_losses_self:
            print(f"Training generator seperate loss at epoch {epoch} is {item / nb_batches_train}")
            gen_epoch_losses.append( item / nb_batches_train )
        stored_gen_seperate.append( gen_epoch_losses )

        print( f"Training generator loss at epoch {epoch} is {generator_losses / nb_batches_train}")
        stored_gen.append(generator_losses / nb_batches_train)

        print( f"Training discriminator loss at epoch {epoch} is {discriminator_losses / (wass_iter*nb_batches_train)}")
        stored_dis.append(discriminator_losses / (wass_iter*nb_batches_train))

        ##### validator loss + take network snapshots
        if( epoch%mod_epochs_valid ) == 0 and epoch >0:
            # fetch network state, create proper name and save ALWAYS
            state_dict = generator.state_dict()
            name = "retargeter_" + str(epoch) + ".zip"
            torch.save(state_dict, name)
            state_dict = discriminator.state_dict()
            name = "discriminator_" + str(epoch) + ".zip"
            torch.save(state_dict, name)

            # save shit
            np.savez("all_losses", losses_disc=stored_dis, losses_gen=stored_gen,
                     losses_gen_seperate=stored_gen_seperate)


        print(f"Elapsed time:{time.time()-start_time}")

    # store the final just in case
    np.savez("all_losses", losses_disc=stored_dis, losses_gen=stored_gen,
             losses_gen_seperate=stored_gen_seperate)






def train_mixamo(configuration):

    print(f"TRAINING CONFIGURATION:{configuration}")

    ## fetch necessary parameters to train
    dataset_file = configuration[ "datasetfile" ]
    nr_layers_gen = configuration[ "nr_layers_gen"]
    nr_heads_gen = configuration[ "nr_heads_gen"]
    model_dim_gen = configuration[ "model_dim_gen"]
    lr_gen = configuration[ "lr_gen"]
    nr_layers_dis = configuration[ "nr_layers_dis"]
    nr_heads_dis = configuration[ "nr_heads_dis"]
    model_dim_dis = configuration[ "model_dim_dis"]
    lr_dis = configuration[ "lr_dis"]
    epochs = configuration[ "epochs"]
    batch_size = configuration[ "batch_size"]
    shuffle_data = configuration[ "shuffle_data"]
    output_folder = configuration[ "output_folder"]
    device = configuration[ "device" ]

    # write also the used configuration
    fo = open('used_configuration.txt', "w")
    for k, v in configuration.items():
        fo.write(str(k) + '=' + str(v) + '\n')
    fo.close()


    ###### create:dataset and dataloader for mixamo dataset
    # character_filter_list = [ 'BigVegas', 'Timmy_m', "Kaya", "Remy_m" ] # for speeding things up
    character_filter_list = None

    # character_filter_list = None
    dataset = RetargetingDataset(dataset_file=dataset_file, filter_list=character_filter_list)
    extra_stuff_dataset = dataset.getExtra()

    # data_loader = DataLoader( dataset, batch_size=batch_size, shuffle=shuffle_data, drop_last=False, pin_memory=True, num_workers=1)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, drop_last=True, pin_memory=False)


    ###### fetch first sample to get dimensions of the motion and the skeleton
    first_sample = dataset[0]
    motions = first_sample[1]
    flat = first_sample[3]
    first_motion = motions[0]
    first_flat = flat[ 0 ]
    seq_len, input_dim = first_motion.shape
    skeleton_dim = first_flat.shape[1]
    output_dim = input_dim

    # total number of classes
    nr_classes = len(motions) + 1


    ##### create retargeting model: generator
    retargeter = RetargeterGeneratorEncoderBoth(nr_layers=nr_layers_gen,
                                            dim_input=input_dim, # motion dimension
                                            dim_model=model_dim_gen,
                                            seq_len=seq_len,
                                            dim_skeleton=skeleton_dim, # skeleton dimension
                                            nr_heads=nr_heads_gen,
                                            dropout=0.1)
    # state_dict = torch.load("../../restore_point/retargeter_510.zip")
    # retargeter.load_state_dict(state_dict)



    ##### create classifier: discriminator
    discriminator = Discriminator(nr_layers =  nr_layers_dis,
                                dim_input = input_dim,
                                dim_model =  model_dim_dis,
                                seq_len = seq_len,
                                dim_skeleton = skeleton_dim,
                                nr_classes = nr_classes,
                                nr_heads = nr_heads_dis,
                                dropout = 0.1 )
    # state_dict = torch.load("../../restore_point/discriminator_510.zip")
    # discriminator.load_state_dict(state_dict)

    #### load validator network: hardcode layers+heads+model_dim: one model to rule them all
    validator = RetargeterGeneratorEncoderBoth(nr_layers=6,
                            dim_input=input_dim,
                            dim_model=96,
                            seq_len=seq_len,
                            dim_skeleton= skeleton_dim,
                            nr_heads=8,
                            dropout=0.1)

    # restore model
    print(f"Loading validator model")
    validator_name = "../../../validation_model/validator_96.zip"
    state_dict = torch.load(validator_name)
    validator.load_state_dict(state_dict)
    mu_sigma_name = "../../../validation_model/mu_sigma_96.npz"
    mu_sigma = np.load( mu_sigma_name )


    # precalculate ee-distances
    global  ee_distances
    if character_filter_list is not None:
        ee_distances = ee_distances[[7,15,16,17]] # [ 'BigVegas', 'Timmy_m', "Kaya", "Remy_m" ]
    h_orig = einops.repeat( ee_distances, 'skel ee -> (skel batch) frames ee xyz', batch=batch_size, frames=seq_len, xyz=3 ).to(device)


    ##### perform training
    train_network( generator=retargeter, discriminator=discriminator,  data_loader=data_loader,
                   device=device, num_epochs=epochs, lr_gen = lr_gen, lr_dis=lr_dis,
                   extra_data=extra_stuff_dataset, validator=validator, mu_sigma=mu_sigma, h_orig=h_orig )

    ###### save the final network
    state_dict = retargeter.state_dict()
    torch.save(state_dict, "retargeter_final.zip")



if __name__ == "__main__":

    start_time = time.time()

    # read configuration file
    print("READING CONFIGURATION PARAMETERS....")
    parameters, configurations = fetch_conf_parameters()

    # move to the trials  folder for results
    print( "MOVING TO TRIALS FOLDER...")
    trials_dir = "trials"
    if not os.path.isdir(trials_dir):
        os.mkdir( trials_dir)
    os.chdir(trials_dir)


    for conf in configurations:

        # move to random folder for saving the results
        save_folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        tmp = os.getcwd() + '/' + save_folder
        os.mkdir(tmp)
        os.chdir( tmp )

        tmp_conf = { item:val for item,val in list(zip(parameters,conf)) }
        train_mixamo( tmp_conf )

        # return back to the trials folder
        os.chdir("../")

        # clean cuda memory
        print("CLEANING CUDA MEMORY......")
        torch.cuda.empty_cache()

    os.chdir("../")
    print( f"TOTAL TRAINING ELAPSE TIME:{time.time()-start_time}")


import os

import einops
import torch.utils.data as data
import torch
import torch.nn as nn
import sys

from torch.onnx.symbolic_opset9 import mul

sys.path.append( '../models')
sys.path.append( '../utils')
sys.path.append( '../train')
sys.path.append( '.')
from torch.utils.data import DataLoader
from shutil import copyfile, copy2
from bvh_manipulation import *
from configuration import fetch_conf_parameters, fetch_conf_parameters_noargs
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



if __name__ == "__main__":

    # load validator model
    input_dim = 111,
    seq_len = 30
    validator = RetargeterGeneratorEncoderBoth(nr_layers=6,
                            dim_input=111,
                            dim_model=96,
                            seq_len=seq_len,
                            dim_skeleton = 135,
                            nr_heads=8,
                            dropout=0.1)

    # restore model
    print(f"Loading validator model")
    validator_name = "trials/validation_model/validator_96.zip"
    state_dict = torch.load(validator_name)
    validator.load_state_dict(state_dict)
    validator.eval()

    # fetch mu+sigma
    mu_sigma_name = "trials/validation_model/mu_sigma_96.npz"
    mu_sigma = np.load(mu_sigma_name)
    mu = mu_sigma["mu_real"]
    sigma = mu_sigma["sigma_real"]

    # training dataset
    character_filter_list = None
    dataset = RetargetingDataset(dataset_file="trials/normalized_dataset.txt", filter_list=character_filter_list)
    extra_stuff_dataset = dataset.getExtra()
    data_loader = DataLoader( dataset, batch_size=100, shuffle=False, drop_last=False, pin_memory=False)
    nb_batches_train = len(data_loader)
    print(f"Batches:{nb_batches_train}")

    path = "/home/vlasdas/Desktop/thesis/experiments/"
    path = "/home/vlasdas/Desktop/thesis/best/both"


    for root, dirs, files in os.walk(path, topdown=False):

        print(f"Parsing folder:{root}")
        os.chdir( root )

        print("READING CONFIGURATION PARAMETERS....")
        for name in files:
            if name.endswith(".txt"):
                print(name)
                parameters, conf = fetch_conf_parameters_noargs()
                tmp_conf = {item: val for item, val in list(zip(parameters, conf[0]))}
                nr_layers_gen =tmp_conf['nr_layers_gen']
                model_dim_gen = tmp_conf['model_dim_gen']
                print(f"nr_layers_gen:{nr_layers_gen}, model_dim_gen:{model_dim_gen}")


        for name in files:
            if name.endswith(".zip"):
                # print(name)

                if "best" in name or "final" in name :
                    print(f"Skipping:{name}")
                    continue

                tmp = int(name.split('_')[1].split('.')[0])
                if tmp < 500:
                    continue

                ##### create retargeting model: generator
                retargeter = RetargeterGeneratorEncoderBoth(nr_layers=nr_layers_gen,
                                                            dim_input=111,  # motion dimension
                                                            dim_model=model_dim_gen,
                                                            seq_len=seq_len,
                                                            dim_skeleton=135,  # skeleton dimension
                                                            nr_heads=8,
                                                            dropout=0.1)

                # restore model
                print(f"Loading trained model:{name}")
                state_dict = torch.load(name)
                retargeter.load_state_dict(state_dict)
                retargeter.eval()

                for batch in data_loader:

                    ## unpack info from input batchs
                    motions = batch[1]
                    flat_skeletons = batch[3]

                    ##  concatenate motions and skeletons and move to device
                    input_seq = torch.cat(motions, dim=0).to("cpu")
                    input_skeleton = torch.cat(flat_skeletons, dim=0).to("cpu")

                    # fid = compute_validation_loss(retargeter=retargeter, validator=validator,
                    #                               input_seq=input_seq, input_skeleton=input_skeleton,
                    #                               mu=mu, sigma=sigma)
                    # print(f"FID score:{fid}")

                r = 1







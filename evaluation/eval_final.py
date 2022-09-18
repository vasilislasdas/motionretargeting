import torch
import torch.utils.data as data
import torch
import torch.nn as nn
import sys
sys.path.append( '../models')
sys.path.append( '../utils')
sys.path.append( '../others/dmr/')
sys.path.append( '../others/')
# from retargeterGeneratorEncoder_latent import RetargeterGeneratorEncoderLatent
from retargeterGeneratorEncoderBoth import RetargeterGeneratorEncoderBoth
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader
from shutil import copyfile, copy2
from preprocess import *
from train_final import *
from others.dmr.IK import fix_foot_contact
from bvh_parser import BVH_file
from bvh_writer import BVH_writer

torch.autograd.set_detect_anomaly(False)

def get_height(file):
    file = BVH_file(file)
    return file.get_height()


def evaluate_mixamo():

    print("PREPARING EVALUATION....")

    # fetch dataset contents
    # character_filter_list = ["Olivia_m", "BigVegas", "Claire"]
    # character_filter_list = ['Olivia_m', 'Timmy_m', 'BigVegas', 'Claire' ]
    # character_filter_list = ['Olivia_m', 'Claire']
    character_filter_list = None

    dataset = RetargetingDataset(dataset_file="normalized_dataset.txt", filter_list=character_filter_list)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    extra = dataset.getExtra()
    all_edges = extra["edges"]
    joint_names = extra["joints"]
    skeleton_offsets = extra["offsets"]
    maximum = extra["max"]
    minimum = extra["min"]
    joint_indices = extra["joint_indices"]

    # fetch random sample
    sample = next(iter(data_loader))
    print(f"Random index:{sample[-1]}")

    # characters
    all_characters = sample[0]
    print(f"All characters:{all_characters} ")

    # fetch all motions
    all_motions = sample[1]
    nr_chars = len(all_motions)
    print(f"Number of motions:{len(all_motions)}")

    # fetch motion_types
    all_motion_types = sample[2]
    print(f"All motion types :{all_motion_types} ")

    # fetch flat skeletons
    all_flat_skeletons = sample[3]
    print(f"All flat types :{ len(all_flat_skeletons) } ")

    # fetch all initpositions
    positions = sample[4]

    # get random character
    ind = torch.randint(low=0, high=len(all_characters), size=(1,)).item()
    ind = 3
    print(f"ind:{ind}")

    # fetch motion word and its relevant data
    motion_word = all_motions[ ind ]
    character = all_characters[ind][0]  # name is a tuple.: 0 needed
    motion_type = all_motion_types[ind][0]# motion type is a tuple
    edges = all_edges[ind]
    joints = joint_names[ind]
    source_init_positions = positions[ind]
    flat_skeleton = all_flat_skeletons[ind]

    print( f"Motion:{motion_word.shape}" )
    print( f"Character:{character}" )
    print( f"Motion type:{motion_type}" )
    print(f"Flat skeleton:{flat_skeleton.shape}")

    # retargeting skeleton
    ind_retarget = torch.randint(low=0, high=len(all_characters), size=(1,)).item()
    ind_retarget =7
    print(f"ind_retarget:{ind_retarget}")
    flat_skeleton_retarget = all_flat_skeletons[ind_retarget]
    character_retarget = all_characters[ind_retarget][0]  # name is a tuple.: 0 needed
    edges_retarget = all_edges[ind_retarget]
    joints_retarget = joint_names[ind_retarget]
    target_init_positions = positions[ind_retarget]

    print(f"Retargeting Character:{character_retarget}")


    ######## prepare input output
    input_seq = motion_word
    skeleton = flat_skeleton
    input_skeleton = skeleton
    batch, seq_len, input_dim = input_seq.shape
    target_skeleton = flat_skeleton_retarget


    ##################### read configuration parameters from conf file
    print( "READING CONFIGURATION PARAMETERS...." )
    parameters, configurations = fetch_conf_parameters()
    assert(len(configurations)==1)
    configuration = { item:val for item,val in list(zip( parameters,configurations[0])) }
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



    ##### create retargeting model
    # retargeter = RetargeterGeneratorEncoderLatent(nr_layers=nr_layers_gen,
    retargeter = RetargeterGeneratorEncoderBoth(nr_layers=nr_layers_gen,
                            dim_input=input_dim,
                            dim_model=model_dim_gen,
                            seq_len=seq_len,
                            dim_skeleton= skeleton.shape[-1],
                            nr_heads=nr_heads_gen,
                            dropout=0.1)

    # restore model
    print(f"Loading trained model")
    state_dict = torch.load("retargeter.zip")
    retargeter.load_state_dict(state_dict)



    # set network in evaluation mode
    retargeter.eval()


    # predict output seq from input seq: should be the smae
    prediction,_,_,_ = retargeter( input_seq=input_seq, input_skeleton =input_skeleton, target_skeleton=target_skeleton)

    # show some stuff
    loss = nn.MSELoss()
    rec_loss = loss( input_seq[:,:,:91], prediction[:,:,:91] )
    print(f"Prediction loss:{rec_loss.item()}")


    # visualize prediction + original motion
    visualize_result( source_motion=input_seq.squeeze(), source_character=character,
                      source_motion_type=motion_type,  source_positions=source_init_positions,
                      target_character=character_retarget, target_motion=prediction.squeeze(),
                      target_edges=edges_retarget, target_positions=target_init_positions,
                      target_joints=joints_retarget, minimum=minimum, maximum=maximum, suffix=".bvh")





def visualize_result( source_motion, source_character, source_motion_type, source_positions,
                      target_character, target_motion, target_edges, target_joints, target_positions,
                      minimum, maximum, suffix):


    os.chdir("results")

    ## copy original file from motion word locally from the mixamo folder for easier manipulation
    new_name = copy_file_locally( source_character, source_motion_type, dataset_root = "../../dataset/training_set/")
    height = get_height(new_name)

    # read the motion from the locally copied original file to get original edges and joint names
    original_motion, original_edges, original_names, offset = read_motion_quaternion( new_name, downSample=True)


    # reconstruct motions
    delta = maximum - minimum
    restored_original = ((source_motion+1)/2)  * delta + minimum
    restored_predicted = ((target_motion + 1) / 2) * delta + minimum
    isNan = torch.any(restored_original.isnan())
    isInf = torch.any(torch.isinf(restored_original))
    assert (isNan == False and isInf == False)
    isNan = torch.any(restored_predicted.isnan())
    isInf = torch.any(torch.isinf(restored_predicted))
    assert (isNan == False and isInf == False)
    # restored_original[:, :3] += source_positions
    # restored_predicted[:, :3] += source_positions

    # # now write the restored motion
    reduced_file = new_name.replace(".bvh", "_reduced.bvh")
    write_motion_quaternions_to_bvh_directly(motion=restored_original, edges=original_edges, joint_names=original_names,
                                             bvh_output_file=reduced_file)
    tmp_name1 = target_character + "_" + source_motion_type + "_retarget.bvh"
    write_motion_quaternions_to_bvh_directly(motion=restored_predicted, edges=target_edges, joint_names=target_joints,
                                             bvh_output_file=tmp_name1)


    # use inverse kinematics to fix foot sliding
    # fix_foot_contact( 'result.bvh', 'input.bvh', 'result.bvh', height )
    tmp_name2 = target_character + "_" + source_motion_type + "_retarget_IK.bvh"
    fix_foot_contact( tmp_name1, reduced_file, tmp_name2, height)

if __name__ == "__main__":

    evaluate_mixamo()

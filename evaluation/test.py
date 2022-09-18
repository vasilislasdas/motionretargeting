import random

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
from bvh_manipulation import *
from train_final import *
from others.dmr.IK import fix_foot_contact
from bvh_parser import BVH_file
from bvh_writer import BVH_writer


def get_height_frametime(file):
    file = BVH_file(file)
    return file.get_height(), file.frametime

def normalize_motion(motion, max, min ):
    normalized_motion = 2 * (motion - min) / (max - min) - 1
    normalized_motion[torch.isnan(normalized_motion)] = 0
    normalized_motion[torch.isinf(normalized_motion)] = 0
    isNan = torch.any(normalized_motion.isnan())
    isInf = torch.any(normalized_motion.isinf())
    assert (isNan == False)
    assert (isInf == False)

    return normalized_motion

def denormalize_motion(motion, max, min ):

    delta = max - min
    denorm_motion = ((motion + 1) / 2) * delta + min
    isNan = torch.any( denorm_motion.isnan())
    isInf = torch.any(torch.isinf(denorm_motion))
    assert (isNan == False and isInf == False)

    return denorm_motion

def normalize_skeleton(skel, max, min ):

    skel[::5] = (2 * skel[::5] / 28) - 1
    skel[1::5] = (2 * (skel[1::5] / 28)) - 1

    skel[2::5] = (2 * (skel[2::5] - min) / (max - min)) - 1
    skel[3::5] = (2 * (skel[3::5] - min) / (max - min)) - 1
    skel[4::5] = (2 * (skel[4::5] - min) / (max - min)) - 1

    return skel

def get_flat_skeleton( edges ):

    # change the indexing of skeletons:
    new_edges = reindex_edges(edges)
    flat_skeleton = flatten_skeleton(new_edges)
    if flat_skeleton.shape[1] != 135:  # hack: either 135 or 110: pad 25 by repeating last edge
        flat_skeleton = torch.cat([flat_skeleton, fake_skeleton_structure.view(1, 25)], dim=1)
        assert (flat_skeleton.shape[1] == 135)

    return  flat_skeleton.squeeze()


# fetches a file, copies it locally, extracts motions words + flatten skeleton + normalizes
# returns all motions words + skeleton
def preprocess_file( input_bvh_file, target_bvh_file, groundtruth=None ):


    # fetch motion, edges, joint-names
    motion, edges, joints, skel_offset = read_motion_quaternion(input_bvh_file, downSample=True)
    height, frametime = get_height_frametime(input_bvh_file)
    # print(f"Input character:{len(edges), len(joints)}")

    # compute original positions of character
    positions_orig = calculate_positions_from_raw(motion[None, :, :], edges, torch.Tensor(skel_offset)).squeeze()


    # pad dimensions of motion to 111-dim: this is fixed for mixamo(each motion is either 91 or 111-dimensionsl)
    # motion is padded by repeating the rotations/quaternions of the last edge 5 times
    padded_motion = pad_motion_dims( inp=motion )
    assert (padded_motion.shape[-1] == 111)

    # write in reduced form the original file
    write_motion_quaternions_to_bvh_directly(motion=padded_motion, edges=edges,
                                             joint_names=joints,
                                             bvh_output_file="original.bvh")

    # write the groundtruth data also
    if groundtruth is not None:
        motion_ground, edges_ground, joints_ground, skel_offset_ground = read_motion_quaternion(groundtruth, downSample=True)
        write_motion_quaternions_to_bvh_directly(motion=motion_ground, edges=edges_ground,
                                                 joint_names=joints_ground,
                                                 bvh_output_file="groundtruth.bvh")


    # split into motion words of length 30
    padding = 30
    motion_words = torch.split( padded_motion, padding, dim = 0)
    motion_words = list(motion_words)
    # del motion_words[-1]# delete last one:

    init_positions = []
    counter = 0
    for word in motion_words:
        init_pos = word[0, :3].clone()
        init_positions.append(init_pos)
        word[:, :3] -= init_pos
        name = "original_" + str(counter) + ".bvh"
        write_motion_quaternions_to_bvh_directly(motion=word, edges=edges,
                                                 joint_names=joints,
                                                 bvh_output_file=name)
        counter +=1


    # print( init_positions )

    # get reindexed+flattened skeleton from edges
    skel = get_flat_skeleton( edges )

    # dataloader to fetch shit
    dataset = RetargetingDataset(dataset_file="../trials/normalized_dataset.txt", filter_list=['Olivia_m'])
    extra = dataset.getExtra()
    max_motion = extra["max"]
    min_motion = extra["min"]
    max_skel = extra["max_skels"]
    min_skel = extra["min_skels"]

    # normalize motions
    norm_motion_words = []
    for word in motion_words:
        tmp_word = normalize_motion( word, max_motion, min_motion)
        norm_motion_words.append(tmp_word)

    # normalize edges
    skel = normalize_skeleton(skel, max_skel, min_skel )

    ## create and load model
    # retargeter = RetargeterGeneratorEncoderLatent(nr_layers=nr_layers_gen,
    input_dim = 111
    nr_layers = 6
    model_dim = 96
    seq_len = 30
    skel_dim = 135
    retargeter = RetargeterGeneratorEncoderBoth(nr_layers=nr_layers,
                            dim_input=input_dim,
                            dim_model=model_dim,
                            seq_len=seq_len,
                            dim_skeleton= 135,
                            nr_heads=8,
                            dropout=0.1)

    # restore model
    print(f"Loading trained model")
    state_dict = torch.load("../retargeter.zip")
    retargeter.load_state_dict(state_dict)

    # set network in evaluation mode
    retargeter.eval()

    # fetch edges, joint-names from target skeleton
    _, edges_target, joints_target, offsets_target = read_motion_quaternion(target_bvh_file, downSample=False)
    target_skel = get_flat_skeleton(edges_target)
    target_skel = normalize_skeleton(target_skel, max_skel, min_skel )
    target_skel = target_skel[None,None, :]

    # retarget each motion words
    retarget_words = []
    input_skel = skel[ None, None,: ]
    # target_skel = input_skel.clone()
    for word in norm_motion_words:
        # r = 1
        prediction,_,_,_ = retargeter( input_seq=word[None,:,:], input_skeleton = input_skel, target_skeleton=target_skel)
        retarget_words.append( prediction.squeeze() )

    final_words = []
    counter=0
    for i in range(len(init_positions)):
        pos = init_positions[i]
        word = retarget_words[i]
        tmp_word = denormalize_motion(word, max_motion, min_motion)
        name = str(counter) + ".bvh"
        write_motion_quaternions_to_bvh_directly(motion=tmp_word, edges=edges_target,
                                                 joint_names=joints_target,
                                                 bvh_output_file=name)
        tmp_word[:, :3] += pos
        final_words.append(tmp_word)
        counter += 1

    retargeting = torch.cat( final_words, dim = 0 )
    write_motion_quaternions_to_bvh_directly(motion=retargeting, edges=edges_target,
                                             joint_names=joints_target,
                                             bvh_output_file="retarget.bvh")
    exit()
    # fix_foot_contact( "retarget.bvh", "original.bvh", "retarget_IK.bvh", height)

    # calculate positions of feet for retargeted_IK
    # tmp_motion, _,_,_ = read_motion_quaternion("retarget_IK.bvh", downSample=False)
    positions = calculate_positions_from_raw(retargeting[None, :, :], edges_target,
                                             torch.Tensor(offsets_target)).squeeze()
    # positions = calculate_positions_from_raw(tmp_motion[None,:,:], edges_target, torch.Tensor(offsets_target) ).squeeze()
    if len(edges_target) == 22:
        left_foot = positions[0,4,:]
        right_foot = positions[0,8,:] # x z y
    else:
        left_foot = positions[0, 4, :]
        right_foot = positions[0, 9, :]

    # print(f"Left foot:{left_foot}, Right foot:{right_foot}")
    # offset
    dz_ret = min(left_foot[1].item(), right_foot[1].item())
    # dz = (left_foot[1].item() + right_foot[1].item())/2

    # do the same for the orignial motion
    if len(edges) == 22:
        left_foot = positions_orig[0, 4, :]
        right_foot = positions_orig[0, 8, :]  # x z y
    else:
        left_foot = positions_orig[0, 4, :]
        right_foot = positions_orig[0, 9, :]
    # print(f"Left foot:{left_foot}, Right foot:{right_foot}")
    dz_orig = min(left_foot[1].item(), right_foot[1].item())

    # get difference between original and retargeted
    # print(f"Orig:{dz_orig}, retarget:{dz_ret}, diff:{dz_orig - dz_ret}")
    dz = dz_orig - dz_ret

    # shift complete motion
    # shifted = tmp_motion.clone()
    shifted = retargeting.clone()
    if dz_ret < dz_orig:
        shifted[:, 1] += abs(dz)
    else:
        shifted[:, 1] -= abs(dz)


    write_motion_quaternions_to_bvh_directly(motion=shifted, edges=edges_target,
                                             joint_names=joints_target,
                                             bvh_output_file="final.bvh")

    # fix_foot_contact( "final.bvh", input_bvh_file, "final_IK.bvh", height)
    fix_foot_contact("final.bvh", "original.bvh", "final_IK.bvh", height)

    print( "Intra-class retargeting!") if len(edges) == len(edges_target) else print( "Cross-class retargeting!")



def get_character(path):

    character_data = {}

    for root, dirs, files in os.walk(path, topdown=False):

        # print(f"Parsing folder:{root}")
        character = root.split('/')[-1]

        motions = []
        for name in files:
            if name.endswith(".bvh"):
                # print(name)
                motions.append(name)

        character_data[character] = motions
    del character_data['']

    source_char = random.choice(list(character_data.keys()))
    # source_char = "Mousey_m"
    # source_char = "BigVegas"
    source_motion = random.choice( list( character_data[source_char] ) )
    # source_motion = 'Freehang Drop.bvh'
    target_char = random.choice(list(character_data.keys()))
    # target_char = "BigVegas"
    # target_char = "Mousey_m"
    target_char = "Aj"
    target_motion = random.choice(list(character_data[target_char]))
    print(f"Source character+motion:{source_char, source_motion}, Target character:{target_char}")

    source_char = path + source_char + '/'
    target_char = path + target_char + '/'

    return  source_char+source_motion, target_char+target_motion, target_char+source_motion




if __name__ == "__main__":

    os.chdir("test")

    input_file,target_file, groundtruth = get_character("/home/vlasdas/Desktop/thesis/code/dataset/test_set/")


    # # input character + input motion
    # input_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/BigVegas/"
    # input_motion = "Mutant Jump Attack.bvh"
    # input_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/Timmy_m/"
    # input_motion = "Yelling.bvh"
    # input_motion = "Big Jump.bvh"
    # # input_motion = "House Dancing.bvh"
    # # input_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/Pearl_m/"
    # # input_motion = "Samba Dancing (1).bvh"
    # input_file = input_char + input_motion

    # # target character + random file from character to fetch the motion
    # target_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/Kaya/"
    # target_motion = "Inspecting.bvh"
    # target_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/Pearl_m/"
    # target_motion = "Sitting Laughing.bvh"
    # target_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/Timmy_m/"
    # target_motion = "Yelling.bvh"
    # target_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/SportyGranny/"
    # target_motion = "Swinging.bvh"
    # target_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/BigVegas/"
    # target_motion = "Mutant Jump Attack.bvh"
    # target_char = "/home/vlasdas/Desktop/thesis/code/dataset/release_bvh/Remy_m/"
    # target_motion = "Shooting.bvh"    #
    # target_file = target_char+target_motion
    # groundtruth = None

    preprocess_file(input_file, target_file, groundtruth )


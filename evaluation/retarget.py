import random
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import sys
sys.path.append( '../models')
sys.path.append( '../utils')
sys.path.append( '../others/dmr/')
sys.path.append( '../others/')
sys.path.append( '../train_wgan/')
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
from scipy.signal import savgol_filter, general_gaussian
from tqdm import tqdm

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
    # new_edges = edges
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
    for word in motion_words:
        init_pos = word[0, :3].clone()
        init_positions.append(init_pos)
        word[:, :3] -= init_pos


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
        # print( f"starting positions of input:{word[0,:3]}")
        prediction,_,_,_ = retargeter( input_seq=word[None,:,:], input_skeleton = input_skel, target_skeleton=target_skel)
        retarget_words.append( prediction.squeeze() )

    final_words = []
    for i in range(len(init_positions)):
        pos = init_positions[i]
        word = retarget_words[i]
        tmp_word = denormalize_motion(word, max_motion, min_motion)
        tmp_word[:, :3] += pos

        print(f"starting positions of input:{motion_words[i][0, :3]}")
        final_words.append(tmp_word)

    retargeting = torch.cat( final_words, dim = 0 )
    write_motion_quaternions_to_bvh_directly(motion=retargeting, edges=edges_target,
                                             joint_names=joints_target,
                                             bvh_output_file="retarget.bvh")


    # calculate positions of feet for retargeted_IK
    positions = calculate_positions_from_raw(retargeting[None, :, :], edges_target,
                                             torch.Tensor(offsets_target)).squeeze()
    # positions = calculate_positions_from_raw(tmp_motion[None,:,:], edges_target, torch.Tensor(offsets_target) ).squeeze()
    root_ret = positions[0,0,:]
    if len(edges_target) == 22:
        left_foot = positions[0,4,:]
        right_foot = positions[0,8,:] # x z y
    else:
        left_foot = positions[0, 4, :]
        right_foot = positions[0, 9, :]

    # offset
    dz_ret = min(left_foot[1].item(), right_foot[1].item())
    # dz = (left_foot[1].item() + right_foot[1].item())/2

    # do the same for the orignial motion
    root_orig = positions_orig[0, 0, :]
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

    # correction: shift retargeting root node in the root of the original motion
    droot = root_orig - root_ret
    shifted_root = retargeting.clone()
    shifted_root[:, :3] += droot
    write_motion_quaternions_to_bvh_directly(motion=shifted_root, edges=edges_target,
                                             joint_names=joints_target,
                                             bvh_output_file="shifter_root.bvh")

    # # shift complete motion
    # shifted = retargeting.clone()
    shifted = shifted_root.clone()
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


# fetches a file, copies it locally, extracts motions words + flatten skeleton + normalizes
# returns all motions words + skeleton
def preprocess_file_v2( input_bvh_file, target_bvh_file, groundtruth=None ):


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

    # dataloader to fetch stuff
    dataset = RetargetingDataset( dataset_file="../../train_wgan/normalized_dataset.txt", filter_list=['Olivia_m'] )
    extra = dataset.getExtra()
    max_motion = extra["max"]
    min_motion = extra["min"]
    max_skel = extra["max_skels"]
    min_skel = extra["min_skels"]

    # get reindexed+flattened skeleton from edges + normalize skeleton
    source_skel = get_flat_skeleton( edges )
    source_skel = normalize_skeleton(source_skel, max_skel, min_skel )


    assert( padded_motion.shape[0] >= 30)
    print( f"Sequence lenght:{padded_motion.shape[0]}" )
    start = 0
    end = 30
    shift = 29
    motion_words = []

    # create new_start_end
    while end < motion.shape[0]:
        # print(f"Start:{start}, end:{end}")
        tmp_word = padded_motion[start:end].clone()
        motion_words.append(tmp_word)
        start+= shift
        end += shift

    # do it also for the final remaining word if any...
    # print(f"Start:{start}, end:{padded_motion.shape[0]}")
    tmp_word = padded_motion[start:].clone()
    motion_words.append(tmp_word)


    init_positions = []
    centered_words = []
    for word in motion_words:
        init_pos = word[0, :3].clone()
        init_positions.append(init_pos)
        tmp_word = word.clone()
        tmp_word[:, :3] -= init_pos
        centered_words.append(tmp_word)

    # print( init_positions )

    # # test
    # counter = 0
    # for word in centered_words:
    #     print(f"Initial positions of centered motion:{word[0,:3]}")
    #     name = str(counter) + ".bvh"
    #     write_motion_quaternions_to_bvh_directly(motion=word, edges=edges,
    #                                              joint_names=joints,
    #                                              bvh_output_file=name)
    #     counter += 1
    # counter = 10
    # for word in motion_words:
    #     print(f"Initial positions of NON-centered motion:{word[0,:3]}")
    #     name = str(counter) + ".bvh"
    #     write_motion_quaternions_to_bvh_directly(motion=word, edges=edges,
    #                                              joint_names=joints,
    #                                              bvh_output_file=name)
    #     counter += 1


    # normalize motions
    norm_motion_words = []
    for word in centered_words:
        tmp_word = normalize_motion( word, max_motion, min_motion)
        norm_motion_words.append(tmp_word)

    # # shift back the denormalized motion words and compare with original
    # shifted_words = []
    # for i in range(len(init_positions)):
    #     pos = init_positions[i]
    #     word = norm_motion_words[i]
    #     tmp_word = denormalize_motion(word, max_motion, min_motion)
    #     tmp_word[:, :3] += pos
    #     shifted_words.append(tmp_word)
    #
    # counter = 20
    # for word in shifted_words:
    #     name = str(counter) + ".bvh"
    #     write_motion_quaternions_to_bvh_directly(motion=word, edges=edges,
    #                                              joint_names=joints,
    #                                              bvh_output_file=name)
    #     counter += 1

    # put motions back and and write the restored motion
    # reconstructed_motion = torch.cat(shifted_words, dim=0)
    # write_motion_quaternions_to_bvh_directly(motion=reconstructed_motion, edges=edges,
    #                                              joint_names=joints,
    #                                              bvh_output_file="reconstructed_original.bvh")



    ## create and load model
    input_dim = 111
    nr_layers = 6
    model_dim = 96
    seq_len = 30
    skel_dim = 135
    retargeter = RetargeterGeneratorEncoderBoth(nr_layers=nr_layers,
                            dim_input=input_dim,
                            dim_model=model_dim,
                            seq_len=seq_len,
                            dim_skeleton= skel_dim,
                            nr_heads=8,
                            dropout=0.1)

    # restore model
    # print(f"Loading trained model")
    state_dict = torch.load("../retargeter.zip")
    retargeter.load_state_dict(state_dict)

    # set network in evaluation mode
    retargeter.eval()


    # fetch edges, joint-names from target skeleton
    input_skel = source_skel[None, None, :]
    _, edges_target, joints_target, offsets_target = read_motion_quaternion(target_bvh_file, downSample=False)
    target_skel = get_flat_skeleton(edges_target)
    target_skel = normalize_skeleton(target_skel, max_skel, min_skel )
    target_skel = target_skel[None,None, :]
    print( "INTRA-class retargeting!") if len(edges) == len(edges_target) else print( "CROSS-class retargeting!")


    retarget_words = []
    for word in norm_motion_words:
        # r = 1
        prediction,_,_,_ = retargeter( input_seq=word[None,:,:], input_skeleton = input_skel, target_skeleton=target_skel)
        retarget_words.append( prediction.squeeze() )

    # denormalize back motion words and force-shift to 0,0
    final_words = []
    counter = 30
    for word in retarget_words:
        tmp_word = denormalize_motion(word, max_motion, min_motion)
        tmp_pos = tmp_word[0, :3].clone()
        tmp_word[:, :3] -= tmp_pos
        final_words.append(tmp_word)
        name = str(counter) + ".bvh"
        # write_motion_quaternions_to_bvh_directly(motion=tmp_word, edges=edges_target,
        #                                          joint_names=joints_target,
        #                                          bvh_output_file=name)
        counter += 1



    # link words together
    for i in range(1,len(final_words)):
        tmp_word1 = final_words[i-1]
        tmp_word2 = final_words[i]
        last_pos = tmp_word1[-1, :3].clone()
        # print(f"Final position of {i-1}-th word:{last_pos}")
        # print(f"Initial position of {i}-word before:{tmp_word2[0,:3]}")
        # print(f"Final position of {i}-word before:{tmp_word2[-1, :3]}")
        tmp_word2[:, :3] += last_pos
        # print(f"Initial position of {i}-word after:{tmp_word2[0, :3]}")
        # print(f"Final position of {i}-word after:{tmp_word2[-1, :3]} \n")

    # for word in final_words:
    #     print(f"Initial position of {i}-word :{word[0, :3]}")
    #     print(f"Final position of {i}-word:{word[-1, :3]} \n")

    # drop last frame
    for i in range(len(final_words)-1):
        final_words[i] = final_words[i][:-1, :]
        r = 1

    # for word in final_words:
    #     print(f"Initial position of {i}-word :{word[0, :3]}")
    #     print(f"Final position of {i}-word:{word[-1, :3]} \n")

    retargeting = torch.cat( final_words, dim = 0 )
    write_motion_quaternions_to_bvh_directly(motion=retargeting, edges=edges_target,
                                             joint_names=joints_target,
                                             bvh_output_file="retarget.bvh")

    # smooth the result
    smooth = retargeting.clone().detach().numpy()
    # for  i in range(3, smooth.shape[1]):
    #     signal = smooth[:, i]
    #     signal2 = savgol_filter(signal, 5, 2)
    #     smooth[:, i] = signal2
    #     r = 1

    smooth = torch.Tensor(smooth)
    # shift smooth to the root of the original motion
    smooth[:, :3] += motion[0, :3]
    write_motion_quaternions_to_bvh_directly(motion=smooth, edges=edges_target,
                                             joint_names=joints_target,
                                             bvh_output_file="smooth.bvh")

    # fix_foot_contact("smooth.bvh", "original.bvh", "smooth_IK.bvh", height)
    #  fix_foot_contact( "retarget.bvh", "original.bvh", "retarget_IK.bvh", height)

    # calculate positions of feet for retargeted_IK
    # tmp_motion, _,_,_ = read_motion_quaternion("smooth_IK.bvh", downSample=False)
    positions = calculate_positions_from_raw(smooth[None, :, :], edges_target,
                                              torch.Tensor(offsets_target)).squeeze()
    # positions = calculate_positions_from_raw(tmp_motion[None,:,:], edges_target, torch.Tensor(offsets_target) ).squeeze()
    if len(edges_target) == 22:
        left_foot = positions[0,4,:]
        right_foot = positions[0,8,:] # x z y
    else:
        left_foot = positions[0, 4, :]
        right_foot = positions[0, 9, :]

    # # offset
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
    shifted = smooth.clone()
    if dz_ret < dz_orig:
        shifted[:, 1] += abs(dz)
    else:
        shifted[:, 1] -= abs(dz)

    write_motion_quaternions_to_bvh_directly(motion=shifted, edges=edges_target,
                                             joint_names=joints_target,
                                             bvh_output_file="final.bvh")

    print("Fixing foot contact with inverse kinematics")
    source_name = "final.bvh"
    # target_name = "final_IK.bvh"
    ik_iters = 3
    for i in tqdm(range(ik_iters)):
        fix_foot_contact(source_name, "original.bvh", source_name, height)
        # source_name = target_name

    # cleanup intermediate files
    os.remove( "retarget.bvh")
    os.remove( "smooth.bvh")

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
    source_motion = random.choice( list( character_data[source_char] ) )
    target_char = random.choice(list(character_data.keys()))
    target_motion = random.choice(list(character_data[target_char]))
    # print(f"Source character+motion:{source_char, source_motion}, Target character:{target_char}")
    print(f"Source character:{source_char}")
    print(f"Source motion   :{source_motion}")
    print(f"Target character:{target_char}")

    source_char = path + source_char + '/'
    target_char = path + target_char + '/'

    return  source_char+source_motion, target_char+target_motion, target_char+source_motion




if __name__ == "__main__":

    ret_results = "results"
    if not os.path.isdir(ret_results):
        os.mkdir( ret_results)
    os.chdir(ret_results)

    # Retargeting can be done in either the test-set, or in the original training set
    # Since the training set contains unique motions from every character, the result is also unique(no grountruth)
    test_set = True
    if test_set:
        input_file, target_file, groundtruth = get_character("../../dataset/test_set/")
        print(f"Input file    :{os.path.abspath( input_file ) }")
        print(f"Groundtruth   :{os.path.abspath(groundtruth) }")
        # Note: Target file is a random file from the target character used to read the skeleton info.
        # Note: Any bvh file from the target skeleton can be used.
        print(f"Target file   :{os.path.abspath(target_file)}")

        # perform the retargeting
        preprocess_file_v2(input_file, target_file, groundtruth)

    else:
        input_file, target_file, groundtruth = get_character("../../dataset/training_set/")
        print(f"Input file    :{os.path.abspath(input_file)}")
        print(f"Groundtruth   :no groundtruth exists!!")
        print(f"Target file   :{os.path.abspath(target_file)}")
        preprocess_file_v2(input_file, target_file, None)



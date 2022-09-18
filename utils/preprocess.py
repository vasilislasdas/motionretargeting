import os
import sys
import time

import einops
import numpy
import torch

sys.path.append('../others/dmr')
sys.path.append('../others/')

from bvh_parser import BVH_file
from bvh_writer import BVH_writer
from collections import Counter
from einops import repeat
import pickle
from shutil import copyfile
from random import choices
from operator import itemgetter
from dmr.Kinematics import ForwardKinematics
from Kinematics import ForwardKinematics
from dataclasses import dataclass

torch.set_printoptions(sci_mode=False)

character_dict = {
    'Olivia_m': 0,
    'Paladin_m': 1,
    'Michelle_m': 2,
    'Pearl_m': 3,
    'Ortiz_m': 4,
    'Pumpkinhulk_m': 5,
    'Abe_m': 6,
    'BigVegas': 7,
    'Jasper_m': 8,
    'LolaB_m': 9,
    'Knight_m': 10,
    'SportyGranny': 11,
    'Joe_m': 12,
    'Maria_m': 13,
    'James_m': 14,
    'Timmy_m': 15,
    'Kaya': 16,
    'Remy_m': 17,
    'Malcolm_m': 18,
    'Claire': 19,
    'Aj': 20,
    'Yaku_m': 21,
    'ParasiteLStarkie_m': 22,
    'Racer_m': 23,
    'Liam_m': 24}

character_shifts = {'Olivia_m': 4,
                    'Paladin_m': 3,
                    'Michelle_m': 3,
                    'Pearl_m': 4,
                    'Ortiz_m': 3,
                    'Pumpkinhulk_m': 4,
                    'Abe_m': 3,
                    'BigVegas': 15,
                    'Jasper_m': 4,
                    'LolaB_m': 4,
                    'Knight_m': 5,
                    'SportyGranny': 14,
                    'Joe_m': 4,
                    'Maria_m': 4,
                    'James_m': 3,
                    'Timmy_m': 3,
                    'Kaya': 15,
                    'Remy_m': 3,
                    'Malcolm_m': 4,
                    'Claire': 15,
                    'Aj': 14,
                    'Yaku_m': 3,
                    'ParasiteLStarkie_m': 4,
                    'Racer_m': 4,
                    'Liam_m': 3}

dict_cat1 = {0: 2, 1: 12, 2: 7, 3: 4, 4: 10, 5: 11, 6: 24, 7: 19, 8: 16, 9: 22, 10: 23, 11: 25, 12: 26, 13: 27, 14: 28,
             15: 13, 16: 0, 17: 1, 18: 8, 19: 9, 20: 3, 21: 5, 22: 6, 23: 20, 24: 21, 25: 15, 26: 17, 27: 18}
dict_cat2 = {0: 14, 1: 12, 2: 7, 3: 4, 4: 10, 5: 24, 6: 19, 7: 16, 8: 22, 9: 2, 10: 25, 11: 26, 12: 28, 13: 13, 14: 0,
             15: 8, 16: 3, 17: 5, 18: 6, 19: 20, 20: 15, 21: 17, 22: 18}


@dataclass
class FakeArgs:
    fk_world = True
    pos_repr = '3d'
    rotation = 'quaternion'


args = FakeArgs()

fake_skeleton_structure = torch.cat([torch.arange(22, 27)[:, None], torch.arange(23, 28)[:, None]], dim=1)
fake_skeleton_structure = torch.cat([fake_skeleton_structure, torch.zeros(5, 3)], dim=1)
fake_skeleton_structure = torch.tensor([0, 9, 0, 0, 0] * 5)


def reindex_edges(edges):
    new_edges = [None] * len(edges)
    dict_cat = None
    if len(edges) == 22:
        dict_cat = dict_cat2
    else:
        dict_cat = dict_cat1

    for i, sublist in enumerate(edges):
        joint1 = sublist[0]
        joint2 = sublist[1]
        dist = sublist[2]
        new_joint1 = dict_cat[joint1]
        new_joint2 = dict_cat[joint2]
        new_sub = []
        new_sub.append(new_joint1)
        new_sub.append(new_joint2)
        new_sub.append(dist)
        new_edges[i] = new_sub
        r = 1

    # test
    for a, b in zip(edges, new_edges):
        a = numpy.array(a[2])
        b = numpy.array(b[2])
        assert (numpy.all(a == b))

    return new_edges


def flatten_skeleton(skeleton):
    # flat_list = [item for sublist in skeleton for item in sublist]
    result = []
    for sublist in skeleton:
        result.extend(sublist[0:2])
        tmp = [item for item in sublist[2]]
        result.extend(tmp)

    return torch.Tensor(result).float()[None, :]


# copy file in local folder
def copy_file_locally(character, motion_type, dataset_root="../../dataset/release_bvh/"):
    original_file = dataset_root + character + "/" + motion_type
    dst_file = character + '_' + motion_type.replace(" ", "")
    copyfile(original_file, dst_file)

    return dst_file


def create_motion_words_from_sample(motion, window_length, shift):
    # get frame number
    frames = motion.shape[0]

    # start and end
    start = 0
    end = window_length
    motion_words = []

    while end <= frames:
        # print(f"Start:{start},end:{end}")

        # fetch sub-motion
        word = motion[start:end, :]
        # print( f"Init position:{word[0,:2]}")
        # print(f"Word dimensions:{word.shape}")
        isNan = torch.any(word.isnan())
        assert (isNan == False)

        # store new sub_motion
        motion_words.append(word.clone())

        # incement sliding window start/end
        start += shift
        end += shift

    return motion_words


def get_dataset_statistics(dataset_root_dir="../dataset/training_set/"):
    # store total duration of each character
    character_duration = {}

    ### iterate the filesystem
    for root, dirs, files in os.walk(dataset_root_dir, topdown=False):

        if root == dataset_root_dir:
            continue

        # print(f"Parsing folder:{root}")
        character = root.split('/')[-1]
        print(f"Parsing character:{character}")

        # total duration of motion for each character
        total_duration = 0

        for name in files:

            if name.endswith(".bvh"):
                # create name of bvh file to process
                bvh_file = os.path.join(root, name)
                # print(f"Parsing bvh file:{bvh_file}")

                # fetch motion etc to get frametime
                motion, edges, joints, offsets = read_motion_quaternion(bvh_file, downSample=True)
                frames = motion.shape[0]
                # print(f"Frames:{frames}")
                total_duration += frames
                r = 1

        # store the total duration
        character_duration[character] = total_duration

    print(f"Character duration:{character_duration}")


def create_and_preprocess_dataset(dataset_root_dir="../dataset/training_set/",
                                  normalization=True,
                                  downsampling=False,
                                  window_length=60,
                                  shift=30):
    # test if folder exist
    if not os.path.exists(dataset_root_dir):
        print("Training folder does not exist!! EXITING...")
        exit(0)

    # stored the character data
    character_data = {}
    maximum = None
    minimum = None

    number_motion_words_per_character = {}

    ### iterate the filesystem
    for root, dirs, files in os.walk(dataset_root_dir, topdown=False):

        # skip root folder
        if root == dataset_root_dir:
            r = 1
            print("test")
            continue

        print(f"Parsing folder:{root}")
        character = root.split('/')[-1]

        # per character data
        motions = []
        motion_types = []
        skeleton = None  # stored only once per character
        joint_names = None  # stored only once per character
        flat_skeleton = None  # stored only once per character
        skeleton_offsets = None  # stored only once per character
        initial_positions = []  # for every motion word keep its original x,y

        # variable that makes sure that we store only once character-specific data
        stored = False

        number_motion_words_per_character[character] = 0

        for name in files:

            if name.endswith(".bvh"):

                # create name of bvh file to process
                bvh_file = os.path.join(root, name)
                print(f"Parsing bvh file:{bvh_file}")

                # fetch motion, edges, joint-names
                motion, edges, joints, skel_offset = read_motion_quaternion(bvh_file, downSample=downsampling)

                # pad dimensions of motion to 111-dim: this is fixed for mixamo(each motion is either 91 or 111-dimensionsl)
                # motion is padded by repeating the rotations/quaternions of the last edge 5 times
                padded_motion = pad_motion_dims(inp=motion)
                assert (padded_motion.shape[-1] == 111)

                # break motion into motion words with adaptive shift
                shift = character_shifts[character]
                motion_words = create_motion_words_from_sample(padded_motion, window_length=window_length, shift=shift)

                # TODO: experimental: make each motion 0-based
                init_positions = []
                for word in motion_words:
                    init_pos = word[0, :3].clone()
                    init_positions.append(init_pos)
                    # print(f"init pose before:{word[0, :2]}")
                    word[:, :3] -= init_pos
                    # print(f"init pose after :{word[0, :2]}")
                    r = 1
                initial_positions.extend(init_positions)
                # print( init_positions )

                # store motion
                motions.extend(motion_words)
                len_words = len(motion_words)

                # store per character duration for testing and statistics
                number_motion_words_per_character[character] += len_words

                # store motion type
                motion_types.extend([name] * len_words)

                if character == "BigVegas":
                    r = 1

                # store only once per character
                if not stored:
                    # store edges
                    skeleton = edges

                    # store joint names
                    joint_names = joints

                    # store offsets needed for FK
                    skeleton_offsets = torch.tensor(skel_offset).float()

                    # change the indexing of skeletons: TODO experimental
                    new_edges = reindex_edges(edges)

                    # pad and store also flattened skeleton
                    flat_skeleton = flatten_skeleton(new_edges)
                    if flat_skeleton.shape[1] != 135:  # hack: either 135 or 110: pad 25 by repeating last edge
                        flat_skeleton = torch.cat([flat_skeleton, fake_skeleton_structure.view(1, 25)], dim=1)
                        assert (flat_skeleton.shape[1] == 135)

                    # mark as stored the data
                    stored = True

        r = 1
        tmp = [motions, motion_types, skeleton, joint_names, flat_skeleton, skeleton_offsets, initial_positions]
        character_data[character] = tmp

    print(f"Number of motion words per character:{number_motion_words_per_character}")

    ###### resample with replacement to have balanced classes
    max_motions = max(list(number_motion_words_per_character.values()))
    print(f"Maximum number of words:{max_motions}")

    # for each character repeat as many motions and motion types as needed + compute zero-based position max-min
    tmp_all_joints = []
    for key, value in character_data.items():

        # fetch motion and motion types
        mot = value[0]
        print(f"LENGTH MOTION:{len(mot)}")
        types = value[1]
        pos = value[-1]
        tmp_edges = value[2]
        tmp_offsets = value[5]

        # test if there are is nan or isinf
        for item in mot:
            cond = torch.any(torch.isnan(item))
            assert (cond == False)
            cond = torch.any(torch.isinf(item))
            assert (cond == False)

        # nr of motions words
        nr_words = len(mot)

        print(f"Character:{key}, motion words:{nr_words}")

        # extra samples
        extra = max_motions - nr_words
        if extra == 0:
            print("Nothing to add...continuing...")
            continue

        # sample with replacement indices
        new_indices = choices(list(range(nr_words)), k=extra)

        # create motion copies
        new_copies = [mot[index].clone() for index in new_indices]
        assert (extra == len(new_copies))
        value[0].extend(new_copies)

        # create motion type copies
        new_copies = None
        new_copies = [types[index] for index in new_indices]
        assert (extra == len(new_copies))
        value[1].extend(new_copies)

        # create initial positions copies
        new_copies = None
        new_copies = [pos[index].clone() for index in new_indices]
        assert (extra == len(new_copies))
        value[-1].extend(new_copies)

        # put things back
        character_data[key] = value

    # verify the resampling process
    for key, value in character_data.items():
        print(f"Character:{key}, motion words:{len(value[0])}")

    ####### normalize between -1,1
    if normalization:
        print("Normalizing motion+skeletal data...")

        # # concatenate all motions to compute min,max: but before compute
        motions = []
        for key, value in character_data.items():
            motions.extend(value[0].copy())

        # concatenate motions to compute min max
        motions = torch.cat(motions, dim=0)
        isNan = torch.any(motions.isnan())
        assert (isNan == False)

        # find min,max
        minimum = torch.amin(motions, dim=0)
        maximum = torch.amax(motions, dim=0)

        # iterate again and normalize motion and  for each character
        tmp_skels = []
        for key, value in character_data.items():
            # normalize motion
            mot = torch.cat(value[0])
            # assert( not torch.any(minimum==maximum))
            normalized_motion = 2 * (mot - minimum) / (maximum - minimum) - 1
            normalized_motion[torch.isnan(normalized_motion)] = 0
            normalized_motion[torch.isinf(normalized_motion)] = 0
            isNan = torch.any(normalized_motion.isnan())
            isInf = torch.any(normalized_motion.isinf())
            assert (isNan == False)
            assert (isInf == False)
            value[0] = einops.rearrange(normalized_motion, '(batches frames) dims -> batches frames dims',
                                        frames=window_length)

            # normalize flattened skeleton to -1 1
            flat_skel = value[4]
            tmp_skels.append(flat_skel)

            # put back the normalized data
            character_data[key] = value


        # # handle skeletons globally
        cat_skels = torch.cat(tmp_skels)
        min_skels = torch.amin(cat_skels.view(-1))
        max_skels = torch.amax(cat_skels.view(-1))
        for i in range(cat_skels.shape[0]):
            skel = cat_skels[i]
            skel[::5] = (2 * skel[::5] / 28) - 1
            skel[1::5] = (2 * (skel[1::5] / 28)) - 1
            skel[2::5] = (2 * (skel[2::5] - min_skels) / (max_skels - min_skels)) - 1
            skel[3::5] = (2 * (skel[3::5] - min_skels) / (max_skels - min_skels)) - 1
            skel[4::5] = (2 * (skel[4::5] - min_skels) / (max_skels - min_skels)) - 1

        norm_skels = cat_skels
        counter = 0
        for key, value in character_data.items():
            # operate only on the skeletons
            value[4] = norm_skels[counter:counter + 1]
            character_data[key] = value
            counter += 1

    # create the dict to store all the info
    print("Saving data...")

    # store also max and min of motion + max_min of all positions + skeletons
    character_data["maximum"] = maximum
    character_data["minimum"] = minimum
    character_data["min_skels"] = min_skels
    character_data["max_skels"] = max_skels

    # store the file locally and save it in the training folder
    out_txt = "normalized_dataset.txt"
    with open(out_txt, "wb") as fp:  # Pickling        ...
        pickle.dump(character_data, fp)

    # zipping data for faster uploading in colab
    # print("Zipping data...")
    # out_zip = "normalized_dataset.zip"
    # command = "zip " + out_zip + " " + out_txt
    # os.system(command)

    # move it in the training folder
    os.rename(out_txt, "../train_wgan/" + out_txt)


# creates a 20 vector of the positions replicated. 3X6 xyz + 1X xy
def creating_padding_vector(posistions):
    # repeat 7 times the positions and discard the last one to get to 20
    repeated_pos = repeat(posistions, 'frames pos ->  frames ( h pos )', h=7)
    repeated_pos = repeated_pos[:, :-1]
    # print( repeated_pos.shape )
    return repeated_pos


# all mixamo skeletons after reduced joints have dims either 111 or 91 with the first 3 dims being positions
# repeat the the last the quaternions of the last edge(4 last rows) 5 times
def pad_motion_dims(inp):
    # fetch dims
    a, motion_dims = inp.shape

    if motion_dims == 111:  # do nothing
        return inp  # do nothing
    else:  # fetch last 4 rows and repeat 5 times
        last_rotations = inp[:, -4:]
        last_rotations = torch.zeros_like(last_rotations)
        last_rotations[:, 0] = 1
        padding = einops.repeat(last_rotations, " a b -> a (m b)", m=5)
        new_inp = torch.cat((inp, padding), 1)
        return new_inp


# reads motion data from file in the form of quaternions+root node position over time
# returns  (nr_frames) X 3(xyz of root)) + (4x ( all_joints_nr - 1), edges(structure), joint names
# first xyz and then the quaternions
def read_motion_quaternion(file_path=None, downSample=False):
    assert (file_path is not None)

    # read the file to extract info
    file = BVH_file(file_path)

    # extracted motion is rotations(jointsx4) + XYZ of root node X frames
    motion = file.to_tensor(quater=True)

    # we can always downsample
    if downSample:
        motion = motion[:, ::2]  # take every second frame
        length = motion.shape[-1]  # get length
        length = length // 2 * 2  # get proper length
        motion = motion[..., :length]  # fetch motion with proper length

    # transpose motion to be of the form: frames X dims
    motion = torch.transpose(motion, 0, 1)

    # put the positions in front and the quaternions after
    positions = motion[:, -3:]
    motion = torch.cat([positions, motion], dim=1)
    motion = motion[:, :-3]

    return motion, file.edges, file.names, file.offset


# motion: quaternions
# bvh_input_file: bvh file with the skeleton for which we want to write motion data: has to be of the same character
# bvh_output_file: output bvh
# Note: we can get the edges and filenames from any bvh file for a character, we just insert new motion data in it
def write_motion_quaternions_to_bvh(motion, bvh_input_file, bvh_output_file):
    # read the input file to get all info
    file = BVH_file(bvh_input_file)  # fetch the edges from the skeleton we want to write data to
    writer = BVH_writer(file.edges, file.names)

    # call the auxillary function
    write_motion_quaternions_to_bvh_directly(motion, file.edges, file.names, bvh_output_file)


def write_motion_quaternions_to_bvh_directly(motion, edges, joint_names, bvh_output_file):
    # create writer with edges and joint_names
    writer = BVH_writer(edges, joint_names)

    # this is a hack due to the nature of the returned motion when reading quaternions
    # need to restore the original dimensions of the motion + put positions at the end
    positions = motion[:, 0:3]

    # change the motion-dim if needed
    if len(edges) == 22:
        # print("motion was just hacked")
        motion = motion[:, 0:91]  # this is the dim with 22 joints

    # put positions at the end
    motion = torch.cat([motion, positions], dim=1)
    motion = motion[:, 3:]

    # since we return the transpose of motion, for the write we transpose the motion before
    writer.write_raw(motion.t(), 'quaternion', bvh_output_file)


def get_unique_numbers(numbers):
    unique = []

    for number in numbers:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique


def read_dataset_from_disk(datasetfile="../utils/normalized_dataset.txt"):
    with open(datasetfile, "rb") as fp:  # Unpickling
        dataset = pickle.load(fp)

    return dataset


def read_dataset_contents(datasetfile="../utils/normalized_dataset.txt"):
    dataset = read_dataset_from_disk(datasetfile=datasetfile)
    motions = dataset["motions"]
    characters = dataset["characters"]
    motion_types = dataset["motion_types"]
    skeletons = dataset["skeletons"]
    joint_names = dataset["joints"]
    maximum = dataset["max"]
    minimum = dataset["minimum"]
    window_length = dataset["window_length"]
    shift = dataset["shift"]
    flattened_skeletons = dataset["flattened_skeletons"]
    minimum_skeletons = dataset["minimum_skeletons"]
    maximum_skeletons = dataset["maximum_skeletons"]

    return motions, characters, motion_types, skeletons, joint_names, maximum, minimum, window_length, shift, \
           flattened_skeletons, minimum_skeletons, maximum_skeletons


def fetch_all_motion_words_from_character_motion_type(all_motions, all_characters, all_motion_types, index,
                                                      window_length):
    # get character and motion
    character = all_characters[index]
    motion_type = all_motion_types[index]
    print(f"character:{character}")
    print(f"motion_type:{motion_type}")

    # get indices of all clips from the same character
    same_characters = [index for index, elem in enumerate(all_characters) if elem == character]

    # get the subset of those that have same motion
    same_motion = [elem for elem in same_characters if all_motion_types[elem] == motion_type]
    print(f"same motion length:{len(same_motion)}")

    # TODO: i use a hack here since my shift is always half of the window_length
    if len(same_motion) == 1:
        print("Only one!")
        return all_motions[index]
    elif (len(same_motion) % 2) == 0:
        print("Even length!")
        motion_words = all_motions[same_motion[::2]]
        last = all_motions[same_motion[-1]]
        last = last[int(window_length / 2):]
        tmp1, tmp2, _ = motion_words.shape
        motion_words = motion_words.reshape(tmp1 * tmp2, -1)
        motion_words = torch.cat((motion_words, last), dim=0)
        return motion_words
    else:
        print("odd length!")
        motion_words = all_motions[same_motion[::2]]
        tmp1, tmp2, _ = motion_words.shape
        motion_words = motion_words.reshape(tmp1 * tmp2, -1)
        return motion_words


def create_one_hot_from_name(keys):
    classes = [character_dict[key] for key in keys]
    # one_hot_vectors = torch.nn.functional.one_hot(torch.tensor(classes), len(character_dict))


# calculate positions: input can be a batch or a pure motions
def calculate_positions_from_raw(raw_motion, edges, offsets):
    assert (raw_motion.ndim == 3)
    # print(f"shape motion:{raw_motion.shape}")
    # print(f"shape edges:{len(edges)}")
    # print(f"shape offset:{offsets.shape}")

    # TODO: hardcoded hack
    if len(edges) == 22 and raw_motion.shape[-1] == 111:
        # print( "HACKED CHARACTER DETECTED!")
        raw_motion = raw_motion[:, :, :91]

    # print(f"shape motion after:{raw_motion.shape}\n")

    # print(f"rawmotion:{raw_motion.shape}")
    # print(f"edges:{len(edges)}")
    # print(f"offset:{offsets.shape, offsets.device}, ")

    # create fk with fake args
    fk = ForwardKinematics(args, edges)

    # put positions at the end to restore format that he requires
    positions = raw_motion[:, :, 0:3]
    motion = torch.cat([raw_motion, positions], dim=2)
    motion = motion[:, :, 3:]

    # reorder motion to dims X frames
    motion = torch.permute(motion, [0, 2, 1])

    # compute positions and stack xyz
    # start = time.time()
    joint_positions = fk.forward_from_raw(motion, offsets)

    return joint_positions


if __name__ == "__main__":

    # get dataset statistics for total number of frames
    # get_dataset_statistics()

    # preprocess data and create dataset to be used for training
    start_time = time.time()
    # create_and_preprocess_dataset(dataset_root_dir="../dataset/training_fake", downsampling=True, window_length=30,
    #                               shift=15)

    create_and_preprocess_dataset(dataset_root_dir="../dataset/training_set", downsampling=True, window_length=30,
                                  shift=15)
    print(f"ELapsed tims:{time.time()-start_time}")

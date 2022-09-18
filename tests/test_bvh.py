import os
import sys
sys.path.append( '../utils')
sys.path.append( '../others')
from shutil import copyfile, copy2
import torch
from dataclasses import dataclass
from bvh_manipulation import *
from dmr.bvh_parser import BVH_file
from dmr.bvh_writer import BVH_writer
from matplotlib import pyplot as plt
import numpy as np
from dmr.Kinematics import ForwardKinematics
from Kinematics import ForwardKinematics
dataset_root_dir = "../dataset/release_bvh/"
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader

def print_3d_skeleton( points, frame, edges ):
    print(plt.rcParams['backend'])

    # plt.ion()

    ax = plt.axes(projection='3d')

    points = points[frame,:,:]
    x = points[:,0].numpy()
    y = points[:,2].numpy()
    z = points[:,1].numpy()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(x, y, z)
    for edge in edges:
        a,b = edge[ 0 ], edge[1]
        r = 1
        tmpx = [ x[a], x[b] ]
        tmpy = [ y[a], y[b] ]
        tmpz = [ z[a], z[b]]
        plt.plot( tmpx, tmpy, tmpz )
    # plt.ylim([0, 50])
    # plt.plot(x,y,z)
    # ax.view_init(0,45)
    plt.show()




@dataclass
class FakeArgs:
    fk_world = True
    pos_repr = '3d'
    rotation = 'quaternion'


def test_calculate_positions_bvh( character = "Aj", motion_type1 = "Belly Dance.bvh", motion_type2 = "Belly Dance.bvh" ):

    # copy the two files locally
    # copy original file locally from the mixamo folder for easier manipulation
    new_name1 = copy_file_locally( character, motion_type1 )
    new_name2 = copy_file_locally(character, motion_type2)

    # read the motion from the locally copied files
    motion1, edges1, names1, offset1 = read_motion_quaternion( new_name1, downSample=False)
    motion2, edges2, names2, offset2 = read_motion_quaternion( new_name2, downSample=False)

    # make offsets to tensors
    offset1 = torch.tensor( offset1, dtype=torch.float32)
    offset2 = torch.tensor( offset2, dtype=torch.float32)

    # destination file
    reduced_file1 = new_name1.replace( ".bvh", "_reduced.bvh" )
    reduced_file2 = new_name2.replace(".bvh", "_reduced.bvh")

    # write both motions to new files locally
    write_motion_quaternions_to_bvh_directly(motion=motion1, edges=edges1, joint_names=names1,
                                             bvh_output_file=reduced_file1)
    write_motion_quaternions_to_bvh_directly(motion=motion2, edges=edges2, joint_names=names2,
                                             bvh_output_file=reduced_file2)

    # make motions have the same numbfer of frames
    frames = min( motion1.shape[0], motion2.shape[0])
    motion1 = motion1[:frames, :]
    motion2 = motion2[:frames, :]

    ### try for forward kinematics model to get positions of joints
    positions1 = calculate_positions_from_raw(raw_motion=motion1[None,:,:], edges=edges1,offsets=offset1)
    positions2 = calculate_positions_from_raw(raw_motion=motion2[None, :, :], edges=edges2, offsets=offset2)
    tmp_pos = torch.cat( [positions1, positions2], dim=0)
    # combine motions
    combined_motions = torch.stack( [motion1, motion2], dim=0)
    combined_positions = calculate_positions_from_raw(raw_motion=combined_motions, edges=edges2, offsets=offset2)
    assert( torch.all(combined_positions==tmp_pos) )
    print(torch.all(combined_positions==tmp_pos))

    r = 1





# test that reading motion from a mocap file and writing to a new bvh file is correct:
# notice that there is
def test_read_write_bvh( character = "Aj", motion_type = "Belly Dance.bvh"):

    # copy original file locally from the mixamo folder for easier manipulation
    new_name = copy_file_locally( character, motion_type )

    # read the motion from the locally copied original file
    motion, edges, names, offset = read_motion_quaternion( new_name, downSample=False)

    # destination file
    reduced_file = new_name.replace( ".bvh", "_reduced.bvh" )

    # write motion to new file
    write_motion_quaternions_to_bvh_directly( motion=motion, edges=edges, joint_names=names, bvh_output_file=reduced_file)


# exctract some motion, distort it and writ to a bvh file
def test_defected_motion( character = "Aj", motion_type = "Belly Dance.bvh"):

    # copy original file locally from the mixamo folder for easier manipulation
    new_name = copy_file_locally( character, motion_type )
    motion,edges,joints, offsets = read_motion_quaternion( new_name )

    # distort root position for some frames:
    motion[ 90:100, 0:3 ] = torch.tensor([200,200,200])
    distorted_file = new_name.replace( ".bvh", "_distorted.bvh" )

    # motion, edges, joint_names, bvh_output_file
    write_motion_quaternions_to_bvh_directly( motion=motion, edges=edges, joint_names=joints, bvh_output_file=distorted_file )


# read skeleton one bvh file ,extract the motion from another and combine these things
def test_write_with_other_tpose( character = 'Aj', motion_type1="Aim Pistol.bvh", motion_type2="Burpee.bvh"):


    motion, edges, names = read_motion_quaternion( input_file, downSample=False)
    motion, edges, names = read_motion_quaternion( input_file, downSample=False)
    outfile = "Aj_dual_replica_nodownsampling_other_tpose.bvh"
    write_motion_quaternions_to_bvh( motion, ref_file, outfile )



def test_restore_full_clip(downSample=True):

    # fetch dataset contents: all of them in one batch
    dataset = RetargetingDataset(dataset_file="../utils/normalized_dataset.txt")
    length = dataset.__len__()
    print(f"samples for each character:{length}")
    data_loader = DataLoader(dataset, batch_size=length, shuffle=False, drop_last=True)
    extra = dataset.getExtra()
    all_edges = extra["edges"]
    all_joint_names = extra["joints"]
    skeleton_offsets = extra["offsets"]
    maximum = extra["max"]
    minimum = extra["min"]

    # fetch the complete dataset for all characters in the form of a batch!! neat!!
    sample = next(iter(data_loader))
    print(f"Random index:{sample[-1]}")

    # fetch all characters
    characters = sample[0]
    print(f"All characters:{len(characters)} ")

    # fetch all motions
    all_motions = sample[1]
    nr_chars = len(all_motions )
    print(f"Number of motions:{len(all_motions)}")

    # fetch all motion_types
    all_motion_types = sample[2]
    print(f"All motion types :{len(all_motion_types)} ")

    # fetch all initial positions
    all_initial_positions = sample[4]
    print(f"All init positions :{len(all_initial_positions)} ")

    #### get random character
    ind = torch.randint( low=0, high=len(characters), size= (1,) ).item()
    # ind = 20
    print(f"ind:{ind}")
    character = characters[ind][0] # all entries are the same...get first one
    print(f"selected character:{character}")

    # get all motions from character
    character_motions = all_motions[ ind ]

    # get all motion types from character
    character_motion_types = all_motion_types[ind]

    # get also edges and joint names for the character
    edges = all_edges[ind]
    joints = all_joint_names[ind]

    # get all character init pos
    character_init_positions = all_initial_positions[ ind ]

    # get random motion type and the relevant motion + init positions
    ind_type = torch.randint(low=0, high=length, size=(1,)).item()
    # ind_type = 41
    print(f"ind_type:{ind_type}")
    character_motion_type = character_motion_types[ind_type]
    character_motion = character_motions[ ind_type ]
    print(f"Selected motion type:{character_motion_type}")
    character_motion_type = character_motion_types[ind_type]


    # find indices for all entries that have the same motion type
    entries = []
    for index, item in enumerate(character_motion_types):
        if item == character_motion_type:
            # print(f"Selected:{character_motion_type, item, index}")
            entries.append(index)
    # print(f"Entries before:{len(entries), entries}")

    tmp = []
    tmp.append(entries[0])
    for i in range(1,len(entries)):
        if  (entries[ i ] - entries[ i -1 ]) == 1:
            tmp.append(entries[ i ])
        else:
            break
    entries = tmp
    # print(f"Entries:{len(entries), entries}")

    # get all entries for that motion
    clip = character_motions[entries]
    print(f"Clip:{clip.shape}")

    # get init_positions
    init_positions = character_init_positions[ entries ]

    # find offset used to create these motion words
    final_clip = None
    if clip.shape[0] > 1:
        tmp1 = clip[0][:, 3:]
        tmp2 = clip[1][:, 3:]
        frame2 = tmp2[0]  # pick the first frame from the second
        used_shift = -1
        for shift in range(tmp1.shape[1]):
            frame1 = tmp1[shift]
            if torch.all(frame1 == frame2):
                print(f"Found shift!!:{shift}")
                used_shift = shift
                break

        if used_shift == character_shifts[character]:
            print("YEAHHHHH")
        else:
            print("ERRRRRRRRRRRRRRORRRRRRRRRRr")

        # for each motion word in clip: scale-back and add the positions
        delta = maximum - minimum
        for i in range( clip.shape[0] ):
            tmp_clip = clip[ i ]
            tmp_clip = ((tmp_clip+1)/2  * delta ) + minimum
            pos = init_positions[ i ]
            tmp_clip[:,:3] += pos
            clip[ i ] = tmp_clip
            r = 1



        ## restore clip from shifted motion words
        final_clip = clip[0]
        for index in range(1, clip.shape[0]):
            word = clip[index]
            extra_frames = word[-used_shift:, :]
            final_clip = torch.cat([final_clip, extra_frames], dim=0)
    else:
        final_clip = clip[0]



    ##### reconstruct full bvh from the overlapping windows

    ### restore the motion
    restored_motion = final_clip
    isNan = torch.any(restored_motion.isnan())
    isInf = torch.any(torch.isinf(restored_motion))
    assert (isNan == False and isInf == False)


    ### copy original file from motion word locally from the mixamo folder for easier manipulation
    new_name = copy_file_locally(character, character_motion_type)

    # read the motion from the locally copied original file
    original_motion, original_edges, original_names, original_offset = read_motion_quaternion(new_name,
                                                                                              downSample=downSample)

    # destination file
    reduced_file = new_name.replace(".bvh", "_reduced.bvh")

    # write motion to new file
    write_motion_quaternions_to_bvh_directly(motion=original_motion, edges=original_edges, joint_names=original_names,
                                             bvh_output_file=reduced_file)

    # now write the restored motion
    restored_file = reduced_file.replace(".bvh", "_restored.bvh")
    print(f"restored_motion.shape:{restored_motion.shape}")
    write_motion_quaternions_to_bvh_directly(motion=restored_motion, edges=edges, joint_names=joints,
                                             bvh_output_file=restored_file)



# TODO: works but needs alignment in framerate:
def test_restore_single_window( downsample=True):

    # fetch dataset contents
    dataset = RetargetingDataset(dataset_file="../utils/normalized_dataset.txt")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    extra = dataset.getExtra()
    all_edges = extra["edges"]
    joint_names = extra["joints"]
    skeleton_offsets = extra["offsets"]
    maximum = extra["max"]
    minimum = extra["min"]
    joint_indices = extra[ "joint_indices"]

    # fetch random sample
    sample = next(iter(data_loader))
    print(f"Random index:{sample[-1]}")

    # characters
    characters = sample[0]
    print(f"All characters:{characters} ")

    # fetch all motions
    all_motions = sample[1]
    nr_chars = len(all_motions )
    print(f"Number of motions:{len(all_motions)}")

    # fetch motion_types
    motion_types = sample[2]
    print(f"All motion types :{motion_types} ")

    # initial positions
    positions = sample[4]

    # get random character
    ind = torch.randint( low=0, high=len(characters), size= (1,) ).item()
    print(f"ind:{ind}")


    # fetch motion word and its relevant data
    motion_word = all_motions[ ind ].squeeze()
    character = characters[ ind ][0] # name is a tuple.: 0 needed
    motion_type = motion_types[ind][0] # motion type is a tuple
    edges = all_edges[ind]
    joints = joint_names[ind]
    init_position = positions[ind]
    print(f"Motion:{motion_word.shape}")
    print(f"Character:{character}")
    print(f"Motion type:{motion_type}")


    ### restore the motion
    print(f"max:{torch.amax(motion_word.view(-1))}")
    print(f"min:{torch.amin(motion_word.view(-1))}")
    delta = maximum - minimum
    restored_motion = ( (motion_word+1)/2  ) * delta + minimum
    isNan = torch.any(restored_motion.isnan())
    isInf = torch.any(torch.isinf(restored_motion))
    assert (isNan == False and isInf == False )

    # scale back the motion word
    restored_motion[ :, :3 ] += init_position


    ### copy original file from motion word locally from the mixamo folder for easier manipulation
    new_name = copy_file_locally(character, motion_type)

    # read the motion from the locally copied original file
    original_motion, original_edges, original_names, original_offset = read_motion_quaternion( new_name, downSample=downsample)

    # destination file
    reduced_file = new_name.replace(".bvh", "_reduced.bvh")

    # write motion to new file
    write_motion_quaternions_to_bvh_directly(motion=original_motion, edges=original_edges, joint_names=original_names,
                                             bvh_output_file=reduced_file)


    # now write the restored motion
    restored_file = reduced_file.replace(".bvh", "_restored.bvh")
    print(f"restored_motion.shape:{restored_motion.shape}")
    write_motion_quaternions_to_bvh_directly(motion=restored_motion, edges=edges, joint_names=joints,
                                             bvh_output_file=restored_file)






if __name__ == "__main__":

    # test_read_write_bvh( character='Olivia_m', motion_type='Clean And Jerk.bvh' )
    # test_read_write_bvh(character='BigVegas', motion_type='Taking Cover.bvh')
    # test_defected_motion( character='Kaya', motion_type='Bellydancing.bvh')
    test_restore_single_window()
    # test_restore_full_clip(downSample=True)
    test_calculate_positions_bvh( character='Olivia_m', motion_type1='Baseball Idle (1).bvh', motion_type2='Baseball Walk Out.bvh' )
    test_calculate_positions_bvh( character='BigVegas', motion_type1='Agony.bvh', motion_type2='Action Idle To Standing Idle.bvh' )




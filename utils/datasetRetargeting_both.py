import time

import einops
import torch
import torch
import torch.nn as nn
import sys
sys.path.append( '../models')
import torchinfo
import tqdm
import pickle
from collections.abc import Iterable   # import directly from collections for Python < 3.3
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from bvh_manipulation import read_dataset_contents
import time
from retargeterGeneratorEncoderOnlyBoth import RetargeterGeneratorEncoderBoth

all_joints = ['RightUpLeg',
 'LeftArm',
 'Three_Arms_split_Hips',
 'LeftLeg',
 'Neck',
 'RightToe_End',
 'RightLeg',
 'RightHand',
 'RightFoot',
 'LeftHand',
 'Head',
 'LeftShoulder',
 'LeftToeBase',
 'RightShoulder',
 'HeadTop_End',
 'Neck1',
 'HipsPrisoner',
 'LeftForeArm',
 'LeftShoulder_split',
 'Spine',
 'Pelvis',
 'Hips',
 'LeftToe_End',
 'Spine1_split',
 'RHipJoint',
 'LeftUpLeg',
 'LHipJoint',
 'RightToeBase',
 'LowerBack',
 'RightShoulder_split',
 'RightHand_split',
 'RightArm',
 'Spine2',
 'LeftHand_split',
 'Spine1',
 'RightForeArm',
 'LeftFoot',
 'Three_Arms_Hips']

skel_category1 = ['Olivia_m',
 'Paladin_m',
 'Michelle_m',
 'Pearl_m',
 'Ortiz_m',
 'Pumpkinhulk_m',
 'Abe_m',
 'Jasper_m',
 'LolaB_m',
 'Knight_m',
 'Joe_m',
 'Maria_m',
 'James_m',
 'Timmy_m',
 'Remy_m',
 'Malcolm_m',
 'Yaku_m',
 'ParasiteLStarkie_m',
 'Racer_m',
 'Liam_m']

skel_category2 = ['BigVegas', 'SportyGranny', 'Kaya', 'Claire', 'Aj']

skel_category1_speedup = ['Olivia_m', 'Timmy_m']
skel_category2_speedup = [ 'BigVegas' ]


class RetargetingDatasetBoth(Dataset):

    def __init__( self, dataset_file = None, filter_list = None ):
        super(RetargetingDatasetBoth, self).__init__()

        assert( dataset_file is not None )
        self.file = dataset_file
        self.filter_list = filter_list

        print(f"Character filter_list:{self.filter_list}")

        # parse dataset and do the magic here
        self._read_dataset()
        self._create_indices()



    def _read_dataset(self):

        with open( self.file, "rb") as fp:  # Unpickling
            dataset = pickle.load(fp)

        # fetch all data

        # store minimum and maximum and delete them from the dictionary to avoid errors
        self.maximum = dataset["maximum"]
        self.minimum = dataset["minimum"]
        del dataset["maximum"]
        del dataset["minimum"]
        self.pos_minimum = dataset["positions_minimum"]
        self.pos_maximum = dataset["positions_maximum"]
        del dataset["positions_minimum"]
        del dataset["positions_maximum"]



        # dataset length = the number of samples per character
        self.len = dataset["Aj"][0].shape[0]
        self.dataset = dataset

        # set active characters that are used
        if self.filter_list is None:
            self.active_characters = len(list(self.dataset.keys()))
        else:
            self.active_characters = len(self.filter_list)


        # pre-save, motions, characters, motion_type, flat_skeleton for all motions
        #[ motions, motion_types, skeleton, joint_names, flat_skeleton, skeleton_offsets, initial_positions ]
        self.motions = []
        self.characters = []
        self.motion_types = []
        self.flat_skeletons = []
        self.edges = []
        self.skeleton_offsets = []
        self.joint_names = []
        self.initial_positions = []


        for key, value in self.dataset.items():

            if self.filter_list is not None and (key not in self.filter_list):
                continue

            # print( f"Parsing character:{key}")

            # characters
            self.characters.append( key )

            # motions: fetch only the correct motion: ditch aritificial shit
            character_motions = value[0]
            if key in skel_category2:
                character_motions = character_motions[:,:, :91]
                r = 1
            self.motions.append(character_motions)


            # motion types
            character_motion_types = value[1]
            self.motion_types.append(character_motion_types)

            # edges
            self.edges.append(value[2])

            # joint names
            self.joint_names.append(value[3])

            # flat skeletons
            character_flat_skeleton = value[4]
            if key in skel_category2:
                character_flat_skeleton = character_flat_skeleton[ :,:110]
            self.flat_skeletons.append( character_flat_skeleton )

            # offsets
            self.skeleton_offsets.append(value[5])

            # initial positions: keep only the relevant ones
            self.initial_positions.append( value[6] )

        r = 1

    # concatenates the complete list of all joints and for each skeleton creates a 0-1 existence vector
    def _create_indices(self):

        self.joint_indices = []

        for joints in self.joint_names:
            joint_indices = [0] * len(all_joints)
            for index, name in enumerate(all_joints):
                if name in joints:
                    joint_indices[ index ] = 1

            self.joint_indices.append( joint_indices )

        r = 1


    def __len__(self):
        return self.len

    def getExtra(self):

        extra = {}
        extra["characters"] = self.characters
        extra["edges"] = self.edges
        extra["joints"] = self.joint_names
        extra["offsets"] = self.skeleton_offsets
        extra["joint_indices"] = self.joint_indices

        # check if filter_list is in category2, then modify max min:
        if set(self.filter_list).issubset( set( skel_category2 ) ):
            print( "Category 2")
            extra["max"] = self.maximum[:91]
            extra["min"] = self.minimum[:91]
            extra["positions_minimum"] = self.pos_minimum[:69]
            extra["positions_maximum"] = self.pos_maximum[:69]
        else:
            extra["max"] = self.maximum
            extra["min"] = self.minimum
            extra["positions_minimum"] = self.pos_minimum
            extra["positions_maximum"] = self.pos_maximum


        return  extra

    def __getitem__(self, index ):

        # need to fetch only motion and motion based on the index
        motions = []
        motion_types = []
        init_positions = []

        for character in range(self.active_characters ):
            motions.append( self.motions[ character ][index] )
            motion_types.append( self.motion_types[ character ][ index ] )
            init_positions.append( self.initial_positions[character][index])


        # returned order: character, motion, motion type, flat skeletons, index
        return self.characters, motions,  motion_types, self.flat_skeletons, init_positions, index



# Note: the dataloaded class creates batches for each skeleton and puts them side by side

# if __name__ == "__main__":
#
#     batch = 7
#
#     ##### skeletons with only 27 edges
#     character_filter_list = skel_category1_speedup
#     dataset1 = RetargetingDatasetBoth( dataset_file="normalized_dataset.txt", filter_list= character_filter_list )
#     extras1 = dataset1.getExtra()
#     data_loader1 = DataLoader( dataset1, batch_size=batch, shuffle=True, drop_last=True  )
#     for key, value in extras1.items():
#         print(f"Key:{key}")
#         r = 1
#
#
#     ### test to iterate using iter(next): works!
#     # create iterator from dataloader
#     # train_iterator1 = iter(data_loader1)
#     # nb_batches_train = len(data_loader1)
#     # for i in range(nb_batches_train):
#     #     sample = next(iter(train_iterator1))
#     #     print(f"sampled index:{sample[-1]}")
#     #     r = 1
#
#
#     ##### skeletons with 22 edges
#     character_filter_list = skel_category2_speedup
#     dataset2 = RetargetingDatasetBoth(dataset_file="normalized_dataset.txt", filter_list=character_filter_list)
#     extras2 = dataset2.getExtra()
#     data_loader2 = DataLoader( dataset2, batch_size=batch, shuffle=True, drop_last=True )
#     for key, value in extras2.items():
#         print(f"Key:{key}")
#         r = 1
#
#
#     network = RetargeterGeneratorEncoderBoth(nr_layers =  4,
#                            dim_motion1 = 111,
#                            dim_motion2 =  91,
#                            dim_model =  128,
#                            seq_len = 30,
#                            dim_skeleton1 = 135,
#                            dim_skeleton2 = 110,
#                            nr_heads = 8,
#                            dropout = 0.1 )
#
#     start_time = time.time()
#     for sample1, sample2 in zip(data_loader1,data_loader2 ):
#         # print(f"sampled index from FIRST dataset:{sample1[-1]}")
#         # print(f"sampled index from SECOND dataset:{sample2[-1]}")
#
#         motion1 = sample1[1]
#         motion2 = sample2[1]
#
#         # motion1 = sample1[1][0] # pick a character
#         # flat_skeleton1 = sample1[3][0]
#         #
#         # motion2 = sample2[1][0]  # pick a character
#         # flat_skeleton2 = sample2[3][0]
#
#         # # 4 combination of outputs
#         # out,_,_ = network(motion1, flat_skeleton1, flat_skeleton1)
#         # # print(f"Network output:{out.shape}")
#         # out,_,_ = network(motion1, flat_skeleton1, flat_skeleton2)
#         # # print(f"Network output:{out.shape}")
#         # out,_,_ = network(motion2, flat_skeleton2, flat_skeleton2)
#         # # print(f"Network output:{out.shape}")
#         # out,_,_ = network(motion2, flat_skeleton2, flat_skeleton1)
#         # # print(f"Network output:{out.shape}\n")
#         # # input()
#         r=1
#
#
#     print(f"Elapsed time:{time.time() - start_time}")
#


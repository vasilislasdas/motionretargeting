import torch
import sys
sys.path.append( '../models')
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from retargeterGeneratorEncoder_latent import RetargeterGeneratorEncoderLatent
import numpy as np
import einops
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

class RetargetingDataset( Dataset ):

    def __init__( self, dataset_file = None, filter_list = None ):
        super(RetargetingDataset,self).__init__(  )

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
        self.min_skels = dataset["min_skels"]
        self.max_skels = dataset["max_skels"]
        del dataset["min_skels"]
        del dataset["max_skels"]



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

            # motions
            character_motions = value[0]
            self.motions.append( character_motions )
            r = 1

            # motion types
            character_motion_types = value[1]
            self.motion_types.append(character_motion_types)

            # edges
            self.edges.append(value[2])

            # joint names
            self.joint_names.append(value[3])

            # flat skeletons
            character_flat_skeleton = value[4]
            self.flat_skeletons.append( character_flat_skeleton )

            # offsets
            self.skeleton_offsets.append(value[5])

            # initial positions
            self.initial_positions.append( value[6])
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
        extra["max"] = self.maximum
        extra["min"] = self.minimum
        extra["joint_indices"] = self.joint_indices
        extra["min_skels"] = self.min_skels
        extra["max_skels"] = self.max_skels


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
# e:g:
if __name__ == "__main__":

    # for speeding things up
    character_filter_list = ["Olivia_m", "BigVegas", "Claire" ]
    dataset = RetargetingDataset( dataset_file="trials/normalized_dataset.txt", filter_list= character_filter_list )
    # dataset = RetargetingDataset(dataset_file="trials/normalized_dataset.txt", filter_list=None)
    extras = dataset.getExtra()

    print(f"dataset length:{dataset.len}")

    # create dataloader class
    batch = 1
    data_loader = DataLoader( dataset, batch_size=batch, shuffle=False, drop_last=True  )

    test_sample = next(iter(data_loader))

    # #### load validator network: hardcode layers+heads
    # model_dim=96
    # validator = RetargeterGeneratorEncoderLatent(nr_layers=4,
    #                         dim_input=111,
    #                         dim_model=model_dim,
    #                         seq_len=30,
    #                         dim_skeleton= 135,
    #                         nr_heads=8,
    #                         dropout=0.1)
    #
    # # restore model
    # print(f"Loading validator model")
    # validator_name = "trials/validation_models/validator_" + str(model_dim) + ".zip"
    # state_dict = torch.load(validator_name)
    # validator.load_state_dict(state_dict)


    # #### compute length of end effectors
    all_edges = extras["edges"]
    ee_cat1 = torch.LongTensor([0, 4, 9, 16, 22, 27])  # 28 joints +root node: 0
    ee_dist_cat1 = torch.LongTensor([4, 4, 6, 9, 9])  # distance between root and end effectors
    # ee_cat2 = torch.LongTensor([0, 4, 8, 14, 18, 22])  # 23 joints +root node: 0
    # ee_dist_cat2 = torch.Tensor([4, 4, 6, 8, 8])  # distance between root and end effectors
    cat2_edges = [ [0,1,2,3], [4,5,6,7], [8,9,10,11,12,13], [8,9,10,11,14,15,16,17], [8,9,10,11,18,19,20,21] ]
    cat1_edges = [ [0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13, 14,15],
                   [10, 11, 12,13, 17,18,19,20,21],  [10, 11, 12,13, 22,23,24,25,26]]
    # all_edges = [all_edges[7]]
    # counter = -1
    # ee_distances = torch.zeros(25,5)
    # for  edges in all_edges:
    #     counter += 1
    #     print(counter)
    #     if len(edges) == 27 :
    #         print("Cat1")
    #         cat_edges = cat1_edges
    #         r = 1
    #     else:
    #         print("Cat2")
    #         cat_edges = cat2_edges
    #
    #     lengths = []
    #     for i in range(len(cat_edges)):
    #
    #         indices = cat_edges[i]
    #         # print(indices)
    #         total_len = 0
    #
    #         for index in indices:
    #             # print(edges[index])
    #             offsets = torch.Tensor(edges[index][2])
    #             # print(offsets)
    #             nor = torch.linalg.vector_norm(offsets)
    #             # print(nor)
    #             total_len += nor
    #             r = 1
    #         lengths.append( total_len )
    #         print(f"total_len:{total_len}")
    #     ee_distances[counter] = torch.Tensor(lengths)
    #
    # print(ee_distances)
    # tmp = einops.repeat(ee_distances, 'skel ee -> (skel batch) frames ee xyz', batch=3, frames=2, xyz=3)
    # r = 1
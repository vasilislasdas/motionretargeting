import multiheadattention as mha
import torch
from dataclasses import dataclass
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,TensorDataset,ConcatDataset
import random

corps_name_1 = ['Pelvis', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_3 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_boss = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_boss2 = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'Left_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Right_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_cmu = ['Hips', 'LHipJoint', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RHipJoint', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LowerBack', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_monkey = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_three_arms = ['Three_Arms_Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
corps_name_three_arms_split = ['Three_Arms_split_Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHand_split', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHand_split']
corps_name_Prisoner = ['HipsPrisoner', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm']
corps_name_mixamo2_m = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End', 'Spine', 'Spine1', 'Spine1_split', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftShoulder', 'LeftShoulder_split', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightShoulder_split', 'RightArm', 'RightForeArm', 'RightHand']
# corps_name_example = ['Root', 'LeftUpLeg', ..., 'LeftToe', 'RightUpLeg', ..., 'RightToe', 'Spine', ..., 'Head', 'LeftShoulder', ..., 'LeftHand', 'RightShoulder', ..., 'RightHand']

"""
2.
Specify five end effectors' name.
Please follow the same order as in 1.
"""
ee_name_1 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_2 = ['LeftToe_End', 'RightToe_End', 'HeadTop_End', 'LeftHand', 'RightHand']
ee_name_3 = ['LeftFoot', 'RightFoot', 'Head', 'LeftHand', 'RightHand']
ee_name_cmu = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_monkey = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_name_three_arms_split = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand_split', 'RightHand_split']
ee_name_Prisoner = ['LeftToe_End', 'RightToe_End', 'HeadTop_End', 'LeftHand', 'RightForeArm']
# ee_name_example = ['LeftToe', 'RightToe', 'Head', 'LeftHand', 'RightHand']



corps_names = [corps_name_1, corps_name_2, corps_name_3, corps_name_cmu, corps_name_monkey, corps_name_boss,
               corps_name_boss, corps_name_three_arms, corps_name_three_arms_split, corps_name_Prisoner, corps_name_mixamo2_m]
ee_names = [ee_name_1, ee_name_2, ee_name_3, ee_name_cmu, ee_name_monkey, ee_name_1, ee_name_1, ee_name_1, ee_name_three_arms_split, ee_name_Prisoner, ee_name_2]

cat1 =['Hips',
 'LeftUpLeg',
 'LeftLeg',
 'LeftFoot',
 'LeftToeBase',
 'LeftToe_End',
 'RightUpLeg',
 'RightLeg',
 'RightFoot',
 'RightToeBase',
 'RightToe_End',
 'Spine',
 'Spine1',
 'Spine1_split',
 'Spine2',
 'Neck',
 'Head',
 'HeadTop_End',
 'LeftShoulder',
 'LeftShoulder_split',
 'LeftArm',
 'LeftForeArm',
 'LeftHand',
 'RightShoulder',
 'RightShoulder_split',
 'RightArm',
 'RightForeArm',
 'RightHand']

cat2 =  ['Pelvis',
 'LeftUpLeg',
 'LeftLeg',
 'LeftFoot',
 'LeftToeBase',
 'RightUpLeg',
 'RightLeg',
 'RightFoot',
 'RightToeBase',
 'Hips',
 'Spine',
 'Spine1',
 'Spine2',
 'Neck',
 'Head',
 'LeftShoulder',
 'LeftArm',
 'LeftForeArm',
 'LeftHand',
 'RightShoulder',
 'RightArm',
 'RightForeArm',
 'RightHand']

# ee_name_cmu = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']
ee_cat1 = [0, 4, 9, 16, 22, 27] # 28 joints +root node: 0
ee_cat2 = [0, 4, 8, 14, 18, 22]  # 23 joints +root node: 0
ee_dist_cat1 = [ 4, 4,6,9,9] # distance between root and end effectors
ee_dist_cat2 = [ 4, 4,6,8,8] # distance between root and end effectors
if __name__ == '__main__':
    ee_cat1 = []
    ind = cat1.index('Hips')
    print(f"Index of Hips in cat1:{ind}")
    ee_cat1.append(ind)
    for joint in ee_name_cmu:
        ind = cat1.index( joint )
        print(f"Index of joint in cat1:{ ind }")
        ee_cat1.append( ind )

    print('\n')
    ee_cat2 = []
    ind = cat2.index('Pelvis')
    print(f"Index of Pelvis in cat2:{ind}")
    ee_cat2.append(ind)
    for joint in ee_name_cmu:
        ind = cat2.index(joint)
        print(f"Index of joint in cat2:{ind }")
        ee_cat2.append( ind )

    print(f"ee_cat1:{ee_cat1}")
    print(f"ee_cat2:{ee_cat2}")

    # combined_cat = list(set(cat1+cat2))
    # combined_cat = sorted(combined_cat)
    # print(combined_cat)
    # dict_cat1 = {}
    # dict_cat2 = {}
    # for index, joint in enumerate(cat1):
    #     new_index = combined_cat.index( joint )
    #     print(f"New index of joint {joint} in combined joints:{new_index}")
    #     dict_cat1[ index ] = new_index
    #     z = combined_cat
    # print('\n')
    # for index, joint in enumerate(cat2):
    #     new_index = combined_cat.index( joint )
    #     print(f"New index of joint {joint} in combined joints:{new_index}")
    #     dict_cat2[ index ] = new_index
    #     z = combined_cat
    #
    # print(dict_cat1)
    # print(dict_cat2)
    # r = 1
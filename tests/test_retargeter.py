import torch
import torch.utils.data as data
import torch
import torch.nn as nn
import sys
sys.path.append( '../models')
sys.path.append( '../utils')
from transformer_models import Retargeter
from datasetRetargeting import RetargetingDataset
from torch.utils.data import DataLoader



if __name__ == "__main__":

    # mixamo dataset
    dataset = RetargetingDataset( dataset_file="normalized_dataset.txt" )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # fetch first sample to get dimensions of the motion
    first_sample = dataset[0]
    motion = first_sample[0]
    skeleton = first_sample[3]
    seq_len, input_dim = motion.shape
    print(f"Motion length:{seq_len}X{input_dim}")
    print(f"Flat skeleton:{skeleton.shape}")
    output_dim = input_dim
    nr_layers = 4
    heads = 4
    seq_len = 30
    model_dim = 128


    # def __init__(self, nr_layers:int, dim_input:int, dim_model: int, dim_output:int = None,  nr_heads: int = 1, dropout : float = 0.1, dim_inter: int = None ):
    retargeter = Retargeter( nr_layers=nr_layers,
                              dim_input=input_dim,
                              dim_model=model_dim,
                              dim_output= output_dim,
                              nr_heads=heads,
                              dropout=0.1 )

    train_iterator = iter(data_loader)
    for index, sample in enumerate(train_iterator):

        if index > 2:
            break

        motion = sample[0]
        skeleton = sample[3]

        # print( f"Index:{index}" )
        # print( f"Motion length:{motion.shape}" )
        # print( f"Character:{sample[1]}" )
        # print( f"Motion type:{sample[2]}" )
        # print(f"Flat skeleton:{sample[3].shape}")
        # print(f"Mean:{sample[-1].shape}")

        # def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor, input_skeleton, output_skeleton):
        retargeter(input_seq=motion, target_seq=motion[:,:-4,:], input_skeleton=skeleton, output_skeleton=skeleton )
        r = 1



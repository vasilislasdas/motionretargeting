import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
datasets = []

for i in range(3):
    datasets.append(TensorDataset(torch.arange(i*10, (i+1)*10)))

dataset = ConcatDataset(datasets)
loader = DataLoader(
    dataset,
    shuffle=False,
    num_workers=0,
    batch_size=2
)

for data in loader:
    print(data)
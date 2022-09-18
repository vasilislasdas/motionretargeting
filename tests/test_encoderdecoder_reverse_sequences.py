import torch.utils.data as data
import torch

class ReverseSequences( data.Dataset ):

    def __init__(self, seq_len, dataset_size ):
        super(ReverseSequences,self).__init__()
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.generate_sequence()

    def generate_sequence(self):
        x = torch.randint(low=0,high=10,size=(self.dataset_size, self.seq_len) )
        y = torch.flip(x, dims=[1])
        self.x = x
        self.y = y

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        return( self.x[ item ], self.y[item])







if __name__ == "__main__":
    # dataset = XORDataset(size=6)
    dataset = ReverseSequences(seq_len=4,dataset_size=10)
    print("Size of dataset:", len(dataset))

    data_loader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    for x, y in data_loader:
        print(x,'   ',y)

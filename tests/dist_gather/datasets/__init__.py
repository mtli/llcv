import numpy as np

from torch.utils.data import Dataset


class RangeDataset(Dataset):
    def __init__(self, args, is_train=False):
        super().__init__()
        self.data = list(range(17))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

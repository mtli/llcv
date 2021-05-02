import numpy as np

from torch.utils.data import Dataset


class RandGen(Dataset):
    def __init__(self, args, is_train=False):
        super().__init__()
        self.classes = 4*[None]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return np.random.rand(1, 2, 3), np.random.randint(4)

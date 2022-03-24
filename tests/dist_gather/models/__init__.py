import torch
import torch.nn as nn

class Identity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_para = nn.Parameter(torch.tensor(0.0))

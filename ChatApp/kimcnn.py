import torch
from torch import nn
import torch.nn.functional as F
from config import CFG


class Net(nn.Module):
    def __init__(self, input_shape):
        """
        input_shape -> tuple (n,c,h,w)
        n = batch size
        c = num channels
        h = height
        w = width(768)
        """
        super().__init__()
        # set default values for conv net
        dropout = CFG.data.dropout
        Ks = CFG.data.Ks
        Co = CFG.data.kernel_num  # number of filters for each conv layer
        D = input_shape[3]

        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, 2)
        self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = [
            self.relu(conv(x)).squeeze(3) for conv in self.convs1
        ]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)
        return x

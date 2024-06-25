import torch
import torch.nn as nn

class Pts_anchor(nn.Module):
    def __init__(self) -> None:
        super(Pts_anchor,self).__init__()

    def forward(self, *args):
        print(args)

class Area_anchor(nn.Module):
    def __init__(self) -> None:
        super(Area_anchor, self).__init__()

class Traj_anchor(nn.Module):
    def __init__(self) -> None:
        super(Traj_anchor, self).__init__()
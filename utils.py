import torch
from torch import nn
import torch.nn.functional as F

def getlinear(input_dim, output_dim, shrink_size = 10, act='relu'):
    """From the paper implementation.

    We create a small sequential network, where we map from input dim to 
    input_dim / shrink_size, then to output

    """
    assert input_dim % shrink_size == 0, "input dim {} can't be divided evenly by provided shrink size {}".format(input_dim, shrink_size)
    if act == 'relu':
        activation = nn.ReLU
    else:
        # currently supports relu only
        activation = nn.ReLU

    return nn.Sequential(nn.Linear(input_dim, input_dim//shrink_size), activation(), nn.Linear() )



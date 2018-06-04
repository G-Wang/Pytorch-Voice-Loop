import os
import numpy as np
from operator import itemgetter

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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

    return nn.Sequential(nn.Linear(input_dim, input_dim//shrink_size), activation(), nn.Linear(input_dim//shrink_size, output_dim))


class VCTKDataSet(Dataset):
    """Custom VCTK dataset as following Facebook Voice Loop's paper

    Custom datasets needs to have __getitem__ and __len__ implemented

    """
    def __init__(self, file_path, single_speaker=False):
        self.file_path = file_path
        self.file_list = os.listdir(file_path)
        # let's figure out speaker id
        if not single_speaker:
            speakers = [f.split("_")[0] for f in self.file_list]
            self.speaker_ids = list(set(speakers))
            self.num_speakers = len(self.speaker_ids)
        else:
            self.num_speakers = 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # result is a dictionary
        result = np.load(self.file_path+self.file_list[index])
        # we want to return the phoneme (x), audio_features (y) and speaker_id (spkr)
        # we will return x, y and spkr as torch tensors
        x = result['phonemes']
        y = result['audio_features']
        spkr = self.speaker_ids.index(self.file_list[index].split("_")[0])

        return x, y, spkr


def my_collate_fn(batch):
    """Given a batch of collate functions. This would be a list of lists.

    batch(list with batch size items, each with 3 items in them, x, y, spkr)

    my collate function will batch all x and y. Note that x is the phoneme input, which ususally
    comes as Long tensors starting at 0, to distinguish it from zero padding, we will add one to
    x before padding.

    """
    # create list
    x_list = [item[0] for item in batch]
    x_len_list = [len(item[0]) for item in batch]
    y_list = [item[1] for item in batch]
    y_len_list = [item[1].shape[0] for item in batch]
    spkr_list = [item[2] for item in batch]
    # zip and sort list by input_length
    sort_item = sorted(zip(x_len_list,x_list, y_list, y_len_list, spkr_list), key=itemgetter(0),reverse=True)
    # unzip
    x_len_list, x_list, y_list, y_len_list, spkr_list = zip(*sort_item)
    # create x_padding and y_padding
    max_x_len = max(x_len_list)
    max_y_len = max(y_len_list)
    x_batch = np.vstack([np.pad(li, (0,max_x_len-len(li)), 'constant') for li in x_list])
    y_batch = np.asarray([np.pad(li, ((0,max_y_len-li.shape[0]),(0,0)), 'constant') for li in y_list])
    
    return torch.from_numpy(x_batch).long(), torch.LongTensor(x_len_list), torch.from_numpy(y_batch), torch.LongTensor(y_len_list), torch.LongTensor(spkr_list)



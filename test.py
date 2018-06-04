import torch
from model import Loop
from hparams import Hparams
from utils import VCTKDataSet, my_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

def no_test_train():
    hp = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Loop(hp, device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("model has {} million parameters".format(model.count_parameters()))


    dataset = VCTKDataSet("data/vctk/numpy_features_valid/")

    loader = DataLoader(dataset, shuffle=False, batch_size=10, drop_last=False, collate_fn = my_collate_fn)

    for data in tqdm(loader):
        text, text_list, target, target_list, spkr = data
        # compute loss
        loss = model.compute_loss_batch((text, text_list), spkr, (target, target_list), teacher_forcing=True)
        print(loss.detach().cpu().numpy())


def test_model():
    hp = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Loop(hp, device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("model has {} million parameters".format(model.count_parameters()))
    dataset = VCTKDataSet("data/vctk/numpy_features_valid/")
    loader = DataLoader(dataset, shuffle=False, batch_size=10, drop_last=False, collate_fn = my_collate_fn)

    for data in tqdm(loader):
        text, text_list, target, target_list, spkr = data
        loss = model.compute_loss_batch((text, text_list), spkr, (target, target_list))
        print(loss.detach().cpu().numpy())
    # forward pass through encoding
    #p_out, s_out = model.encoder.forward((text, text_list), spkr)
    #print(p_out.shape, s_out.shape)
    

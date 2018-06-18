import os
import torch
from model import Loop
from hparams import Hparams
from utils import VCTKDataSet, my_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math


def validate(model, loader):
    with torch.no_grad():
        total_loss = 0.0

        for data in tqdm(loader):
            text, text_list, target, target_list, spkr = data
            loss = model.compute_loss_batch((text, text_list), spkr, (target, target_list), teacher_forcing=False)
            total_loss += float(loss.detach().cpu().numpy())
        
        print("total validation loss: {} \n".format(total_loss))

def load_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model
    


def train():
    hp = Hparams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Loop(hp, device)
    # check if we have checkpoint
    checkpoint_path = "checkpoints/last_model.pwf"
    if os.path.isfile(checkpoint_path):
        print("checkpoint found! loading checkpoint model...")
        model = load_from_checkpoint(model, checkpoint_path)
    else:
        print("no checkpoint found, training from scratch...")

    print("model has {} million parameters...".format(model.count_parameters()))

    # hyper-parameters
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100
    batch_size = 25
    grad_norm = 0.5
    valid_epoch = 2

    # training parameters
    print('loading data...')
    train_data = VCTKDataSet("data/vctk/numpy_features/")
    val_data = VCTKDataSet("data/vctk/numpy_features_valid/")
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False, drop_last=False, collate_fn=my_collate_fn)

    print('initial validation...')
    validate(model, val_loader)

    # actual training loop:
    for ep in tqdm(range(epochs)):
        # initialze loss and dataset
        total_loss = 0
        loader = DataLoader(train_data, shuffle=True, drop_last=False, batch_size=batch_size, collate_fn=my_collate_fn)
        for data in tqdm(loader):
            text, text_list, target, target_list, spkr = data
            loss = model.compute_loss_batch((text, text_list), spkr, (target, target_list), teacher_forcing=True)
            # update
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optim.step()
            # save loss
            total_loss += float(loss.detach().cpu().numpy())

        # if total loss is nan
        if math.isnan(total_loss):
            print('total loss is nan! loading from last checkpoint')
            model = load_from_checkpoint(model, checkpoint_path)
            optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            print("loss is good, saving model...")
            torch.save(model.state_dict(), checkpoint_path)

        
        # print loss after every epoch
        print("epoch: {}, total loss: {}".format(ep, total_loss))
        if ep != 0 and ep % valid_epoch == 0:
            print("validating model...   ")
            validate(model, val_loader)
            # save model after every validation
            torch.save(model.state_dict(), "checkpoints/saved_models/val_model_{0:03d}.pwf".format(ep))


if __name__=="__main__":
    train()
            





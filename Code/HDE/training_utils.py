import pickle
from os.path import exists

import torch

from Code.HDE.hde_glove_stack import HDEGloveStack
from Code.Training import device


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(save_path):
    hde = None
    if exists(save_path):
        try:
            checkpoint = torch.load(save_path)
            hde = checkpoint["model"].to(device)
            optimizer = get_optimizer(hde)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("loading checkpoint model at:", save_path, "with", num_params(hde), "trainable params")
        except Exception as e:
            print(e)
            print("cannot load model at", save_path)
    if hde is None:
        hde = HDEGloveStack(hidden_size=200, embedded_dims=100, num_layers=2).to(device)
        optimizer = get_optimizer(hde)
        print("inited model", repr(hde), "with:", num_params(hde), "trainable params")

    return hde, optimizer


def get_training_data(save_path):
    if exists(save_path):
        filehandler = open(save_path + ".data", 'rb')
        data = pickle.load(filehandler)
        return data

    return {"losses": [], "train_accs": [], "valid_accs": []}

def get_optimizer(model, type="sgd"):
    if type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.001)
    if type == "adamw":
        return torch.optim.AdamW(model.parameters, lr=0.001)
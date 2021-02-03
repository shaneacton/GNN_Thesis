import pickle
from os.path import exists

import torch

from Code.HDE.hde_glove_stack import HDEGloveStack
from Code.Training import device
from Viz.loss_visualiser import visualise_training_data


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
        filehandler.close()
        return data

    return {"losses": [], "train_accs": [], "valid_accs": []}


def get_optimizer(model, type="sgd"):
    if type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.001)
    if type == "adamw":
        return torch.optim.AdamW(model.parameters, lr=0.001)


def plot_training_data(data, save_path, print_loss_every, num_training_examples):
    path = save_path + "_losses.png"
    losses, train_accs, valid_accs = data["losses"], data["train_accs"], data["valid_accs"]

    num_prints = len(losses)
    num_trained_examples = num_prints * print_loss_every
    num_epochs = num_trained_examples / num_training_examples
    epochs = [num_epochs * i/len(losses) for i in range(len(losses))]

    visualise_training_data(losses, accuracies=train_accs, show=False, save_path=path, epochs=epochs, valid_accs=valid_accs)


def save_training_data(data, save_path):
    filehandler = open(save_path + ".data", 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()
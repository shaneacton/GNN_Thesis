import math
import pickle
from os.path import exists

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from Code.Config.config import config
from Code.HDE.hde_glove import HDEGloveStack
from Code.Training import device
from Viz.loss_visualiser import visualise_training_data, get_continuous_epochs


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(save_path, **model_kwargs):
    hde = None
    if exists(save_path):
        try:
            checkpoint = torch.load(save_path)
            hde = checkpoint["model"].to(device)
            optimizer = get_optimizer(hde, type=config.optimizer_type)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("loading checkpoint model", hde.name, "at:", save_path, "e:", hde.last_epoch, "i:", hde.last_example,
                  "with", num_params(hde), "trainable params")
        except Exception as e:
            print(e)
            print("cannot load model at", save_path)
    if hde is None:
        hde = HDEGloveStack(**model_kwargs).to(device)
        optimizer = get_optimizer(hde, type=config.optimizer_type)
        print("inited model", hde.name, repr(hde), "with:", num_params(hde), "trainable params")

    return hde, optimizer


def get_training_data(save_path):
    if exists(save_path):
        filehandler = open(save_path + ".data", 'rb')
        data = pickle.load(filehandler)
        filehandler.close()
        return data

    return {"losses": [], "train_accs": [], "valid_accs": []}


def get_optimizer(model, type="sgd", lr=0.001):
    print("using", type, "optimiser")
    params = (p for p in model.parameters() if p.requires_grad)
    if type == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if type == "adamw":
            return torch.optim.AdamW(params, lr=lr)
    if type == "adam":
        return torch.optim.Adam(params, lr=lr)

    raise Exception("unreckognised optimizer arg: " + type)


def plot_training_data(data, save_path, print_loss_every, num_training_examples):
    path = save_path + "_losses.png"
    losses, train_accs, valid_accs = data["losses"], data["train_accs"], data["valid_accs"]
    epochs = get_continuous_epochs(losses, num_training_examples, print_loss_every)
    # print("got epochs:", epochs)
    visualise_training_data(losses, train_accs, epochs, show=False, save_path=path, valid_accs=valid_accs)


def save_data(data, save_path, suffix=".data"):
    filehandler = open(save_path + suffix, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()


def save_config(cfg, save_path):
    filehandler = open(save_path + ".cfg", 'wb')
    pickle.dump(cfg, filehandler)
    filehandler.close()


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cos_val = math.cos(math.pi * float(num_cycles) * 2.0 * progress)  # -1:1
        return max(0.0, 0.5 * (1.0 + cos_val))  # 0 - 1

    return LambdaLR(optimizer, lr_lambda, last_epoch)
import pickle
from os.path import exists

import torch
from torch.optim.lr_scheduler import LambdaLR

from Code.HDE.gated_hde import GatedHDE
from Code.HDE.hde_bert import HDEBert
from Code.HDE.hde_glove import HDEGlove
from Code.Pooling.hde_pool import HDEPool
from Code.Training import device
from Config.config import conf
from Viz.loss_visualiser import visualise_training_data, get_continuous_epochs


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(save_path, **model_kwargs):
    hde = None
    if exists(save_path):
        try:
            checkpoint = torch.load(save_path)
            hde = checkpoint["model"].to(device)
            optimizer = get_optimizer(hde, type=conf.optimizer_type)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler = get_exponential_schedule_with_warmup(optimizer)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("loading checkpoint model", hde.name, "at:", save_path, "e:", hde.last_epoch, "i:", hde.last_example,
                  "with", num_params(hde), "trainable params")
        except Exception as e:
            print(e)
            print("cannot load model at", save_path)
    if hde is None:
        # hde = HDEGlove(**model_kwargs).to(device)
        # hde = HDEBert(**model_kwargs).to(device)
        # hde = HDEPool(**model_kwargs).to(device)
        hde = GatedHDE(**model_kwargs).to(device)

        optimizer = get_optimizer(hde, type=conf.optimizer_type)
        scheduler = get_exponential_schedule_with_warmup(optimizer)
        print("inited model", hde.name, repr(hde), "with:", num_params(hde), "trainable params")

    return hde, optimizer, scheduler


def get_training_results(save_path):
    if exists(save_path):
        filehandler = open(save_path + ".data", 'rb')
        data = pickle.load(filehandler)
        filehandler.close()
        return data

    return {"losses": [], "train_accs": [], "valid_accs": []}


def get_optimizer(model, type="sgd"):
    print("using", type, "optimiser")
    params = (p for p in model.parameters() if p.requires_grad)
    lr = conf.initial_lr
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


def save_conf(cfg, save_path):
    filehandler = open(save_path + ".cfg", 'wb')
    pickle.dump(cfg, filehandler)
    filehandler.close()


def get_exponential_schedule_with_warmup(optimizer, num_grace_epochs=1, decay_fac=0.9):
    """roughly halves the lr every 7 epochs. at e 50, lr is 200 times lower"""

    def lr_lambda(epoch: float):
        if epoch <= num_grace_epochs:
            return 1
        t = epoch - num_grace_epochs
        # print("e:", epoch, "lr_f:", decay_fac ** t)

        return decay_fac ** t

    return LambdaLR(optimizer, lr_lambda)
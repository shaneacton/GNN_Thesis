from os.path import exists

import torch
from torch_geometric.nn import GATConv, SAGEConv

from Code.HDE.gated_hde import GatedHDE
from Code.HDE.hde_bert import HDEBert
from Code.HDE.hde_glove import HDEGlove
from Code.Pooling.hde_pool import HDEPool

from Code.Training import device
from Code.Training.Utils.training_utils import get_exponential_schedule_with_warmup
from Config.config import conf
from Config.config_utils import load_checkpoint_model_config, save_checkpoint_model_config
from Viz import wandb_utils
from Viz.wandb_utils import use_wandb

MODEL_MAP = {"GatedHDE": GatedHDE, "HDEBert": HDEBert, "HDEGlove": HDEGlove, "HDEPool": HDEPool}
GNN_MAP = {"GATConv": GATConv, "SAGEConv": SAGEConv}


def get_model(save_path, NEW_MODEL_CLASS=None, **model_kwargs):
    if NEW_MODEL_CLASS is None:
        NEW_MODEL_CLASS = MODEL_MAP[conf.model_class]
    hde = None
    if exists(save_path):
        hde, optimizer, scheduler = continue_model(save_path)

    if hde is None:
        hde, optimizer, scheduler = new_model(save_path, NEW_MODEL_CLASS, **model_kwargs)

    return hde, optimizer, scheduler


def continue_model(save_path):
    hde, optimizer, scheduler = None, None, None
    try:
        checkpoint = torch.load(save_path)
        hde = checkpoint["model"].to(device)
        optimizer = get_optimizer(hde, type=conf.optimizer_type)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = get_exponential_schedule_with_warmup(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("loading checkpoint model", hde.name, "at:", save_path, "e:", hde.last_epoch, "i:", hde.last_example,
              "with", num_params(hde), "trainable params")
        cfg = load_checkpoint_model_config(save_path)
        if use_wandb:
            wandb_utils.continue_run(cfg.wandb_id, cfg.model_name)

    except Exception as e:
        print(e)
        print("cannot load model at", save_path)

    return hde, optimizer, scheduler


def new_model(save_path, MODEL_CLASS, **model_kwargs):
    hde = MODEL_CLASS(**model_kwargs).to(device)

    optimizer = get_optimizer(hde, type=conf.optimizer_type)
    scheduler = get_exponential_schedule_with_warmup(optimizer)
    if use_wandb:
        wandb_utils.new_run()
    save_checkpoint_model_config(conf, save_path)
    print("inited model", hde.name, repr(hde), "with:", num_params(hde), "trainable params")
    return hde, optimizer, scheduler


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


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
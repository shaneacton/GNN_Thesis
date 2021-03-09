from os.path import exists

import torch
from torch_geometric.nn import GATConv, SAGEConv

from Checkpoint.checkpoint_utils import get_model_checkpoint_folder, save_json_data, model_config_path, model_path, \
    load_json_data
from Code.HDE.hde_cannon import HDECannon
from Code.HDE.hde_bert import HDEBert
from Code.HDE.hde_glove import HDEGlove
from Code.Pooling.hde_pool import HDEPool
from Code.Training import device

from Code.Training.Utils.training_utils import get_exponential_schedule_with_warmup
from Config.config import conf
from Config.config_utils import load_checkpoint_model_config
from Viz import wandb_utils
from Viz.wandb_utils import use_wandb

MODEL_MAP = {"HDECannon": HDECannon, "HDEBert": HDEBert, "HDEGlove": HDEGlove, "HDEPool": HDEPool}
GNN_MAP = {"GATConv": GATConv, "SAGEConv": SAGEConv}


def get_model(name, MODEL_CLASS=None, **model_kwargs):
    if MODEL_CLASS is None:
        MODEL_CLASS = MODEL_MAP[conf.model_class]
    hde = None
    if exists(model_path(name)):
        hde, optimizer, scheduler = continue_model(name)

    if hde is None:
        hde, optimizer, scheduler = new_model(name, MODEL_CLASS=MODEL_CLASS, **model_kwargs)

    return hde, optimizer, scheduler


def continue_model(name):
    hde, optimizer, scheduler = None, None, None
    try:
        path = model_path(name)
        checkpoint = torch.load(path)
        hde = checkpoint["model"].to(device())
        optimizer = get_optimizer(hde, type=conf.optimizer_type)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = get_exponential_schedule_with_warmup(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("loading checkpoint model", hde.name, "at:", path, "e:", hde.last_epoch, "i:", hde.last_example,
              "with", num_params(hde), "trainable params")
        cfg = load_json_data(model_config_path(name))
        if use_wandb:
            wandb_utils.continue_run(cfg["wandb_id"], cfg["model_name"])

    except Exception as e:
        print(e)
        print("cannot load model at", path)

    return hde, optimizer, scheduler


def new_model(name, MODEL_CLASS=None, **model_kwargs):
    from Config.config import conf
    hde = MODEL_CLASS(**model_kwargs).to(device())

    optimizer = get_optimizer(hde, type=conf.optimizer_type)
    scheduler = get_exponential_schedule_with_warmup(optimizer)
    if use_wandb:
        wandb_utils.new_run()
        from Config.config import conf  # reload conf to reflect new wandb_id

    save_json_data(conf.cfg, model_config_path(name))
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
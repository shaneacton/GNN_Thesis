import os
from os.path import exists

import torch
pytorch_geo_installed = True
try:
    from torch_geometric.nn import GATConv, SAGEConv
    from Code.GNNs.linear_gnn2 import LinearGNN2
    from Code.GNNs.linear_gnn3 import LinearGNN3
    from Code.GNNs.linear_gnn_sage import LinearGNNSAGE
    from Code.GNNs.real_sage import RealSAGE
    from Code.GNNs.sage_core import SAGECore
    from Code.GNNs.linear_gnn import LinearGNN

except ModuleNotFoundError:
    pytorch_geo_installed = False

from Checkpoint.checkpoint_utils import save_json_data, model_config_path, model_path, \
    load_json_data, restore_from_backup_folder, load_status
from Code.Embedding.bert_embedder import BertEmbedder
from Code.GNNs.TransGNNs.transformer_gnn_edge import TransformerGNNEdge
from Code.HDE.hde_model import HDEModel
from Code.HDE.hde_rel import HDERel
from Code.Training import dev
from Code.Training.lamb import Lamb
from Code.Utils.training_utils import get_exponential_schedule_with_warmup, get_training_results
from Config.config import get_config
from Code.Utils import wandb_utils
from Code.Utils.wandb_utils import use_wandb

MODEL_MAP = {
    "HDE": HDEModel,
    "HDERel": HDERel
             }

PYGEO_MAP = {"GATConv": GATConv, "SAGEConv": SAGEConv, "Linear2": LinearGNN2, "Linear3": LinearGNN3,
             "LinearSAGE": LinearGNNSAGE, "RealSAGE": RealSAGE, "SAGECore": SAGECore, "Linear": LinearGNN} \
    if pytorch_geo_installed else {}

GNN_MAP = {"TransformerEdge": TransformerGNNEdge}
GNN_MAP.update(PYGEO_MAP)


def get_model_class(model_class_name=None):
    if model_class_name is None:
        model_class_name = get_config().model_class
    MODEL_CLASS = MODEL_MAP[model_class_name]
    return MODEL_CLASS


def get_model(name, MODEL_CLASS=None, **model_kwargs):
    if MODEL_CLASS is None:
        MODEL_CLASS = get_model_class(get_config().model_class)
    hde = None
    if exists(model_path(name)):
        hde, optimizer, scheduler = continue_model(name)

    if hde is None:
        hde, optimizer, scheduler = new_model(name, MODEL_CLASS=MODEL_CLASS, **model_kwargs)

    return hde, optimizer, scheduler


def continue_model(name, backup=False):
    model, optimizer, scheduler = None, None, None
    try:
        path = model_path(name)
        checkpoint = torch.load(path, map_location=dev())
        model = checkpoint["model"].to(dev())
        if type(model.embedder) == BertEmbedder:
            model.embedder.set_all_params_trainable(False)
        optimizer = get_optimizer(model, type=get_config().optimizer_type)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = get_exponential_schedule_with_warmup(optimizer)

        if "second_optim" in checkpoint:
            model.embedder.set_trainable_params()
            bert_optim = get_optimizer(model.embedder, lr=0.0001)
            bert_optim.load_state_dict(checkpoint["second_optim"])
            optimizer = (optimizer, bert_optim)

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("loading checkpoint model", model.name, "at:", path, "e:", model.last_epoch, "i:", model.last_example,
              "with", num_params(model), "trainable params, on device:", dev(), "process:", os.getpid())
        print(model)

        cfg = load_json_data(model_config_path(name))
        get_config().cfg = cfg
        _ = get_training_results(name, backup=backup)  # can sometimes fail due to file corruption
        _ = load_status(name, backup=backup)  # can sometimes fail due to file corruption
        if use_wandb:
            wandb_utils.continue_run(cfg["wandb_id"], cfg["model_name"])
    except Exception as e:
        print("cannot load model at", path)
        print("load error:", e)
        """we will delete this checkpoint folder as it has corrupted"""

        if not backup:
            print("trying backup")
            restore_from_backup_folder(name)
            return continue_model(name, backup=True)
        raise e

    return model, optimizer, scheduler


def new_model(name, MODEL_CLASS=None, **model_kwargs):
    model = MODEL_CLASS(**model_kwargs).to(dev())

    optimizer = get_optimizer(model, type=get_config().optimizer_type)
    scheduler = get_exponential_schedule_with_warmup(optimizer)

    if type(model.embedder) == BertEmbedder:
        model.embedder.set_trainable_params()
        num_bert_params = num_params(model.embedder)
        if num_bert_params > 0:
            bert_optim = get_optimizer(model.embedder, lr=0.0001)
            print("fine tuning bert embedder with ", num_bert_params, " params")
            optimizer = (optimizer, bert_optim)

    if use_wandb:
        wandb_utils.new_run(get_config().model_name)

    save_json_data(get_config().cfg, model_config_path(name))
    print("inited model", model.name, "with:", num_params(model), "trainable params, on device:", dev(), "process:", os.getpid())
    print(model)
    return model, optimizer, scheduler


def get_optimizer(model, type="sgd", lr=None):
    print("using", type, "optimiser")
    params = (p for p in model.parameters() if p.requires_grad)

    if lr is None:
        lr = get_config().initial_lr
    if type == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if type == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if type == "adam":
        return torch.optim.Adam(params, lr=lr)
    if type == "lamb":
        return Lamb(params, lr=lr)

    raise Exception("unreckognised optimizer arg: " + type)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
import json
import pickle
from os import mkdir
from os.path import join, exists

import torch
from filelock import FileLock

from Checkpoint import CHECKPOINT_FOLDER


def save_model(model, optimizer, scheduler):
    model_save_data = {"model": model, "optimizer_state_dict": optimizer.state_dict(),
                       "scheduler_state_dict": scheduler.state_dict()}
    path = model_path(model.name)
    torch.save(model_save_data, path)


def get_model_checkpoint_folder(name):
    return join(CHECKPOINT_FOLDER, name)


def create_model_checkpoint_folder(name):
    path = get_model_checkpoint_folder(name)
    if exists(path):
        raise Exception("checkpoint folder already exists")
    mkdir(path)
    status = get_new_training_status()
    save_json_data(status, training_status_path(name))


def model_path(name):
    path = get_model_checkpoint_folder(name)
    return join(path, name)


def training_results_path(name):
    path = get_model_checkpoint_folder(name)
    return join(path, name + ".training_data")


def training_status_path(name):
    path = get_model_checkpoint_folder(name)
    return join(path, "status.json")


def model_config_path(name):
    path = get_model_checkpoint_folder(name)
    return join(path, name + ".cfg")


def loss_plot_path(name):
    path = get_model_checkpoint_folder(name)
    return join(path, "losses.png")


def load_status(name):
    path = training_status_path(name)
    with FileLock(path + ".lock"):
        status = load_json_data(path)
    return status


def save_status(name, status):
    path = training_status_path(name)
    with FileLock(path + ".lock"):
        save_json_data(status, path)


def get_new_training_status():
    return {"running": True, "completed_epochs": 0, "finished": False}


def load_json_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def save_binary_data(data, path):
    filehandler = open(path, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()
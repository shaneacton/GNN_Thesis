import json
import pickle
import shutil
from os import mkdir
from os.path import join, exists

import torch
from filelock import FileLock

from Checkpoint import CHECKPOINT_FOLDER


def duplicate_checkpoint_folder(name):
    primary = get_model_checkpoint_folder(name)
    backup = get_backup_model_checkpoint_folder(name)
    if exists(backup):
        shutil.rmtree(backup)
    shutil.copytree(primary, backup)
    set_status_value(name, "running", False, backup=True)


def restore_from_backup_folder(name):
    """delete corrupted checkpoint folder, replace with backup and proceed"""
    primary = get_model_checkpoint_folder(name)
    shutil.rmtree(primary)

    backup = get_backup_model_checkpoint_folder(name)

    shutil.copytree(backup, primary)


def save_model(model, optimizer, scheduler):
    model_save_data = {"model": model, "optimizer_state_dict": optimizer.state_dict(),
                       "scheduler_state_dict": scheduler.state_dict()}
    path = model_path(model.name)
    torch.save(model_save_data, path)


def get_model_checkpoint_folder(name, backup=False):
    if backup:
        return get_backup_model_checkpoint_folder(name)
    return join(CHECKPOINT_FOLDER, name)


def get_backup_model_checkpoint_folder(name):
    checkpoint_folder = get_model_checkpoint_folder(name)
    return checkpoint_folder + "_backup"


def create_model_checkpoint_folder(name):
    path = get_model_checkpoint_folder(name)
    if exists(path):
        raise Exception("checkpoint folder already exists")
    mkdir(path)
    status = get_new_training_status()
    save_json_data(status, training_status_path(name))


def model_path(name, backup=False):
    path = get_model_checkpoint_folder(name, backup=backup)
    return join(path, name)


def training_results_path(name, backup=False):
    path = get_model_checkpoint_folder(name, backup=backup)
    return join(path, name + ".training_data")


def training_status_path(name, backup=False):
    path = get_model_checkpoint_folder(name, backup=backup)
    return join(path, "status.json")


def model_config_path(name):
    path = get_model_checkpoint_folder(name)
    return join(path, name + ".cfg")


def loss_plot_path(name):
    path = get_model_checkpoint_folder(name)
    return join(path, "losses.png")


def set_status_value(model_name, key, value, backup=False):
    status = load_status(model_name, backup=backup)
    status[key] = value
    save_status(model_name, status, backup=backup)


def load_status(name, backup=False):
    path = training_status_path(name, backup=backup)
    with FileLock(path + ".lock"):
        status = load_json_data(path)
    return status


def save_status(name, status, backup=False):
    path = training_status_path(name, backup=backup)
    with FileLock(path + ".lock"):
        save_json_data(status, path)


def get_new_training_status():
    return {"running": True, "completed_epochs": 0, "finished": False}


def load_json_data(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print("error loading json data at:", path)
        raise e
    return data


def save_json_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def save_binary_data(data, path):
    filehandler = open(path, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()
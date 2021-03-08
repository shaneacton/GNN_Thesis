import copy
import json
import pickle
from os.path import join, exists
from typing import Dict
import glob

from Checkpoint.checkpoint_utils import model_config_path
from Config import CONFIG_FOLDER


def get_full_config_path(name):
    if ".json" not in name:
        name += ".json"
    paths = glob.glob(CONFIG_FOLDER + "/*/" + name) + glob.glob(join(CONFIG_FOLDER, name))
    if len(paths) > 1:
        raise Exception("multiple configs with same name:", paths)
    if len(paths) == 0:
        raise Exception("no config found with name:", name)

    path = paths[0]
    return path


def load_config(name, add_model_name=True) -> Dict:
    path = get_full_config_path(name)
    if not exists(path):
        raise Exception("no such config file as:", name, "in HDE/Config/")
    with open(path, "r") as f:
        kwargs = json.load(f)
    # sets the model name to the config files name if not provided
    if add_model_name and name != "base" and "train" not in name:
        if "model_name" not in kwargs:
            kwargs.update({"model_name": name})
    return kwargs


def load_effective_config(name, default):
    original = load_config(name)
    default = load_config(default)
    default.update(original)
    return default


def load_configs(model_cfg_name, train_cfg_name="standard_train"):
    train_kwargs = load_effective_config(train_cfg_name, "standard_train")
    model_kwargs = load_effective_config(model_cfg_name, "base")

    all_kwargs = copy.deepcopy(model_kwargs)
    all_kwargs.update(train_kwargs)

    print("loaded config for model:", model_cfg_name)

    return all_kwargs


def load_checkpoint_model_config(name):
    path = model_config_path(name)
    filehandler = open(path, 'rb')
    cfg = pickle.load(filehandler)
    filehandler.close()
    return cfg
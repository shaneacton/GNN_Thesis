import copy
import json
import pickle
from os.path import join, exists
from typing import Dict
import glob

from Config import CONFIG_FOLDER


def load_config(name) -> Dict:
    if ".json" not in name:
        name += ".json"
    paths = glob.glob(CONFIG_FOLDER + "/*/" + name)
    if len(paths) > 1:
        raise Exception("multiple configs with same name:", paths)

    path = paths[0]
    if not exists(path):
        raise Exception("no such config file as:", name, "in HDE/Config/")
    kwargs = json.load(open(path, "r"))
    return kwargs


def load_configs(model_cfg_name, train_cfg_name="standard_train"):
    train_kwargs = load_config(train_cfg_name)
    if train_cfg_name != "standard_train":
        std_train: Dict = load_config("standard_train")
        std_train.update(train_kwargs)  # the conf args are overwritten
        train_kwargs = std_train

    model_kwargs = load_config(model_cfg_name)
    if model_cfg_name != "base":
        base_conf: Dict = load_config("base")
        base_conf.update(model_kwargs)
        model_kwargs = base_conf

    all_kwargs = copy.deepcopy(model_kwargs)
    all_kwargs.update(train_kwargs)

    print("loaded config for model:", model_cfg_name)

    return all_kwargs


def save_checkpoint_model_config(config, path):
    filehandler = open(path + ".cfg", 'wb')
    pickle.dump(config, filehandler)
    filehandler.close()


def load_checkpoint_model_config(path):
    filehandler = open(path + ".cfg", 'rb')
    cfg = pickle.load(filehandler)
    filehandler.close()
    return cfg
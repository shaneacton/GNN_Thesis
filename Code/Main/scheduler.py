import os
import pathlib
import sys
from os.path import join


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Config'))

from Config.config import Config
from Code.Training.trainer import train_model
from Config import load_checkpoint_model_config

file_path = pathlib.Path(__file__).parent.absolute()
CHECKPOINT_FOLDER = join(file_path, "../HDE/Checkpoint")


def train_config(model_cfg_name=None, train_cfg_name=None):
    """train/continue a model using a model config in HDE/Config"""
    global config

    if model_cfg_name is not None or train_cfg_name is not None:
        names = {"model_cfg_name": model_cfg_name, "train_cfg_name": train_cfg_name}
        overrides = {k: v for k, v in names.items() if v is not None}
        config = Config(**overrides)
    path = join(CHECKPOINT_FOLDER, config.model_name)
    train_model(path)


def continue_model(model_name):
    """
        continue a partly trained model using its name
        you don't have to reference the original config to continue a models training
    """
    path = join(CHECKPOINT_FOLDER, model_name)
    cfg = load_checkpoint_model_config(path)
    train_model(path, **cfg)


if __name__ == "__main__":
    train_config()
    # continue_model("hde")
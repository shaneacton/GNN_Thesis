import os
import pathlib
import sys
from os.path import join


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Config'))

from Code.Training.p_trainer import train_model
from Config import load_checkpoint_model_config
from Config.config import conf

file_path = pathlib.Path(__file__).parent.absolute()
CHECKPOINT_FOLDER = join(file_path, "../HDE/Checkpoint")


def train_config():
    """train/continue a model using a model config in HDE/Config"""
    path = join(CHECKPOINT_FOLDER, conf.model_name)
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
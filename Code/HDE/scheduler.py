import argparse
import os
import pathlib
import sys
from os.path import join

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.HDE.Config import load_configs, load_checkpoint_model_config
from Code.HDE.trainer import train_model

file_path = pathlib.Path(__file__).parent.absolute()
CHECKPOINT_FOLDER = join(file_path, "Checkpoint")


def train_config(model_cfg_name, train_cfg_name="standard_train"):
    """train/continue a model using a model config in HDE/Config"""
    cfg = load_configs(model_cfg_name, train_cfg_name=train_cfg_name)

    if "name" in cfg:
        model_name = cfg["name"]
    else:
        model_name = model_cfg_name

    path = join(CHECKPOINT_FOLDER, model_name)
    train_model(path, **cfg)


def continue_model(model_name):
    """
        continue a partly trained model using its name
        you don't have to reference the original config to continue a models training
    """
    path = join(CHECKPOINT_FOLDER, model_name)
    cfg = load_checkpoint_model_config(path)
    train_model(path, **cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_conf', '-m', help='Hyper params for model', default="base")
    parser.add_argument('--train_conf', '-t', help='Training details and memory budgeting', default="standard_train")

    args = parser.parse_args()

    train_config(args.model_conf, train_cfg_name=args.train_conf)
    # continue_model("hde")
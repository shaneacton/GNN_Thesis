import argparse
import os
import sys
from os.path import join


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(join(dir_path_1, 'Code'))
sys.path.append(join(dir_path_1, 'Config'))
sys.path.append(join(dir_path_1, 'Checkpoint'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_conf', '-m', help='Hyper params for model', default="base")
    parser.add_argument('--train_conf', '-t', help='Training details and memory budgeting', default="standard_train")

    args = parser.parse_args()


    from Code.Main.scheduler import train_config
    train_config(args.model_conf, args.train_conf)
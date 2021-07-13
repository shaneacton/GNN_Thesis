import argparse
import atexit
import os
import sys
import time
from os.path import join


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(join(dir_path_1, 'Code'))
sys.path.append(join(dir_path_1, 'Config'))
sys.path.append(join(dir_path_1, 'Checkpoint'))


from Checkpoint.checkpoint_utils import create_model_checkpoint_folder
from Config.config import set_conf_files

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_conf', '-m', help='Hyper params for model', default="base")
    parser.add_argument('--train_conf', '-t', help='Training details and memory budgeting', default="standard_train")
    parser.add_argument('--debug', '-d', help='Whether or not to run the debug configs - y/n', default="n")

    args = parser.parse_args()
    if args.debug == "y":
        args.model_conf = "debug_model"
        args.train_conf = "debug_train"

    set_conf_files(args.model_conf, args.train_conf)

    from Config.config import conf
    from Code.Main.scheduler import train_config, effective_name

    model_name = effective_name(conf.model_name, 0)
    create_model_checkpoint_folder(model_name, safe_mode=True)

    train_config(args.model_conf, args.train_conf, program_start_time=start_time)
import argparse
import os
import sys
import time
from os.path import join

import torch.multiprocessing as mp

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(join(dir_path_1, 'Code'))
sys.path.append(join(dir_path_1, 'Config'))
sys.path.append(join(dir_path_1, 'Checkpoint'))

from Checkpoint.checkpoint_utils import create_model_checkpoint_folder


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_conf', '-m', help='Hyper params for model', default="base")
    parser.add_argument('--model_conf2', '-m2', help='Hyper params for 2nd model', default="")
    parser.add_argument('--train_conf', '-t', help='Training details and memory budgeting', default="standard_train")
    parser.add_argument('--debug', '-d', help='Whether or not to run the debug configs - y/n', default="n")

    args = parser.parse_args()
    if args.debug == "y":
        args.model_conf = "debug_model"
        args.train_conf = "debug_train"

    from Code.Main.scheduler import train_config, effective_name

    if args.model_conf2:  # a second GPU is available. We will run a second config
        ctx = mp.get_context('spawn')

        kwargs = {"model_conf": args.model_conf2, "train_conf": args.train_conf, "gpu_num": 1,
                  "repeat_num": 0, "program_start_time": start_time, "debug": args.debug == "y"}
        process = ctx.Process(target=train_config, kwargs=kwargs)
        process.start()

    from Config.config import conf

    model_name = effective_name(conf.model_name, 0)
    create_model_checkpoint_folder(model_name, safe_mode=True)

    train_config(args.model_conf, args.train_conf, program_start_time=start_time)
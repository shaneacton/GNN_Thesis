import argparse
import atexit
import os
import sys
from multiprocessing import Process
from os.path import join, exists

import torch
from filelock import FileLock

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(join(dir_path_1, 'Code'))
sys.path.append(join(dir_path_1, 'Config'))
sys.path.append(join(dir_path_1, 'Checkpoint'))

from Config.config import set_conf_files
from Checkpoint.checkpoint_utils import get_model_checkpoint_folder, load_status, create_model_checkpoint_folder, \
    save_status
from Config.config_utils import load_config, load_effective_config

GLOBAL_FILE_LOCK_PATH = "../../Checkpoint/scheduler_lock.lock"


def train_config(model_conf=None, train_conf=None, gpu_num=0, repeat_num=0):
    """
        train/continue a model using a model config in HDE/Config
        this method can be run in parallel by different processes
    """
    from Config.config import conf
    if model_conf is not None or train_conf is not None:
        if model_conf is None:
            model_conf = conf.model_cfg_name
        if train_conf is None:
            train_conf = conf.train_cfg_name

    set_conf_files(model_conf, train_conf)
    from Config.config import conf
    from Code.Training.trainer import train_model
    conf.model_name = effective_name(conf.model_name, repeat_num)
    atexit.register(release_status)
    train_model(conf.model_name, gpu_num=gpu_num)


def release_status():
    from Config.config import conf
    status = load_status(conf.model_name)
    status["running"] = False
    save_status(conf.model_name, status)
    print("setting running=False for", conf.model_name)


def effective_name(name, repeat_num):
    return name + "_" + repr(repeat_num)


def get_next_model_config(schedule, repeat_num=1):
    """
        process safe. Only one scheduler can be deciding a config at a time.

        finds first available model config to run. only once all are running/finished, will the scheduler repeat the list
        picks the config which has completed the fewest epochs
    """

    with FileLock(GLOBAL_FILE_LOCK_PATH):
        """
            no two schedulers can pick configs at the same time
            a scheduler must chose a config, and set it to running before it is finished, and can release the lock
        """
        print("getting next config from schedule:", schedule, "repeat num:", repeat_num)
        all_confs = schedule["model_configs"]
        model_names = {c: effective_name(load_effective_config(c, "base")["model_name"], repeat_num) for c in all_confs}
        started = [conf for conf in all_confs if exists(get_model_checkpoint_folder(model_names[conf]))]
        not_started = [conf for conf in all_confs if conf not in started]
        print("started:", started, "not:", not_started)

        if len(not_started) > 0:
            selected = not_started[0]
            create_model_checkpoint_folder(model_names[selected])
            print("starting conf:", selected)
        else:
            min_epochs = 999999999
            best_candidate = None
            for s in started:
                """find the in-progress run with the fewest epochs completed"""
                status = load_status(model_names[s])
                if not status["running"] and not status["finished"]:
                    """not currently running, but not yet finished"""
                    completed_epochs = status["completed_epochs"]
                    if completed_epochs < min_epochs:
                        min_epochs = completed_epochs
                        best_candidate = s
                selected = best_candidate

            if selected is not None:
                """set the status to running"""
                status = load_status(model_names[selected])
                status["running"] = True
                save_status(model_names[selected], status)
                print("continuing conf:", selected)

    """release lock"""
    if selected is None:
        """no candidate was found, either all are running, or complete"""
        if repeat_num < schedule["num_repeats"]:
            return get_next_model_config(schedule, repeat_num=repeat_num+1)
        else:
            """nothing more to be run"""
            print("all configs running or completed")
            return None, None

    return selected, repeat_num


def continue_schedule(debug=False):
    """reads the schedule, as well as which """
    num_gpus = torch.cuda.device_count()
    if not debug:
        schedule = load_config("schedule", add_model_name=False)
    else:
        schedule = load_config("debug_schedule", add_model_name=False)

    train_conf = schedule["train_config"]
    for gpu_id in range(num_gpus):
        """for each available gpu, spawn off a new process to run the next scheduled config"""
        next_model_conf, repeat_num = get_next_model_config(schedule)
        if next_model_conf is None:
            print("no more configs to run. shutting down scheduler")
            exit()
        print("chosen conf:", next_model_conf)
        if gpu_id == num_gpus -1:
            """is last config to run. can run in master thread"""
            train_config(next_model_conf, train_conf, gpu_id, repeat_num)
        else:
            # spawn process
            kwargs = {"model_conf": next_model_conf, "train_conf": train_conf, "gpu_num": gpu_id}
            process = Process(target=train_config, kwargs=kwargs)
            process.start()
            print("starting new process for", next_model_conf, "on gpu:", gpu_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', help='Whether or not to run the debug configs - y/n', default="base")

    args = parser.parse_args()
    continue_schedule(debug=args.debug == "y")
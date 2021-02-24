import argparse

from Config import load_configs


class Config:

    def __init__(self, model_cfg_name="base", train_cfg_name="standard_train"):
        self.model_cfg_name = model_cfg_name
        self.train_cfg_name = train_cfg_name
        self.cfg = load_configs(model_cfg_name, train_cfg_name=train_cfg_name)

        for k, v in self.cfg.items():
            self.__setattr__(k, v)

        if self.max_context_chars != -1:
            print("truncating contexts to", self.max_context_chars, "chars")


parser = argparse.ArgumentParser()
parser.add_argument('--model_conf', '-m', help='Hyper params for model', default="base")
parser.add_argument('--train_conf', '-t', help='Training details and memory budgeting', default="standard_train")

args = parser.parse_args()

config = Config(args.model_conf, train_cfg_name=args.train_conf)

if __name__ == "__main__":

    config = Config("base", "debug_train")
    print(config.max_visualised_candidates)
    print(config.checkpoint_every)
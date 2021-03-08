from Config.config_utils import load_configs


class Config:

    def __init__(self, model_cfg_name="base", train_cfg_name="standard_train"):
        self.model_cfg_name = model_cfg_name
        self.train_cfg_name = train_cfg_name
        self.cfg = load_configs(model_cfg_name, train_cfg_name=train_cfg_name)

        for k, v in self.cfg.items():
            self.__setattr__(k, v)

        if self.max_context_chars != -1:
            print("truncating contexts to", self.max_context_chars, "chars")


conf = None


def set_conf_files(model_cfg_name="base", train_cfg_name="standard_train"):
    global conf
    conf = Config(model_cfg_name, train_cfg_name)


if __name__ == "__main__":

    conf = Config("base", "debug_train")
    print(conf.max_visualised_candidates)
    print(conf.checkpoint_every)
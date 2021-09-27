from Config.config_utils import load_configs


class Config:

    def __init__(self, model_cfg_name="base", train_cfg_name="standard_train"):
        self.model_cfg_name = model_cfg_name
        self.train_cfg_name = train_cfg_name
        self.cfg = load_configs(model_cfg_name, train_cfg_name=train_cfg_name)
        self.cfg["num_transformer_params"] = -1
        self.cfg["num_coattention_params"] = -1
        self.cfg["num_summariser_params"] = -1

        self.cfg["num_embedding_params"] = -1
        self.cfg["num_gnn_params"] = -1
        self.cfg["num_output_params"] = -1

        self.cfg["num_total_params"] = -1
        self.cfg["clean_model_name"] = ""

        for k, v in self.cfg.items():
            self.__setattr__(k, v)

        if self.max_context_chars != -1:
            print("truncating contexts to", self.max_context_chars, "chars")

        self.run_args = None

    def set(self, att_name, value):
        setattr(self, att_name, value)
        self.cfg[att_name] = value

    @property
    def hidden_size(self):
        if self.use_simple_hde:
            return self.embedded_dims
        return self.embedded_dims * 2

conf = None


def get_config():
    global conf
    if conf is None:
        set_conf_files()
    return conf


def set_conf_files(model_cfg_name="base", train_cfg_name="standard_train"):
    global conf
    conf = Config(model_cfg_name, train_cfg_name)


if __name__ == "__main__":
    conf = Config("base", "debug_train")
    print(conf.max_visualised_candidates)
    print(conf.checkpoint_every)
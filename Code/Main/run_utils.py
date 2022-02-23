from Config.config import set_conf_files


def get_model_config(model_conf=None, train_conf=None, repeat_num=0, run_args=None):
    from Config.config import conf
    if model_conf is not None or train_conf is not None:
        if model_conf is None:
            model_conf = conf.model_cfg_name
        if train_conf is None:
            train_conf = conf.train_cfg_name

    set_conf_files(model_conf, train_conf)
    from Config.config import conf
    model_name = effective_name(conf.model_name, repeat_num)
    conf.set("clean_model_name", conf.model_name)
    conf.set("model_name", model_name)
    conf.run_args = run_args
    if run_args.max_runtime and run_args.max_runtime != -1:
        conf.max_runtime_seconds = int(run_args.max_runtime)
    return conf


def effective_name(name, repeat_num):
    return name + "_" + repr(repeat_num)
from Config.config import conf, get_config

try:
    import wandb
    use_wandb = conf.use_wandb
except Exception as e:
    print("wandb error:", e)
    use_wandb = False


_wandb_run = None


def wandb_run():
    return _wandb_run


def new_run(model_name, config=None):
    global _wandb_run
    global use_wandb
    if config is None:
        config = get_config()
    id = wandb.util.generate_id()
    config.set("wandb_id", id)
    try:
        _wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=config.cfg, resume=True, name=model_name, id=id)
    except Exception as e:
        print("cannot init wandb session. turning off wandb logging")
        print(e)
        use_wandb = False
        config.cfg["wandb_id"] = -1
        config.wandb_id = -1
        return None

    return _wandb_run


def continue_run(id, model_name=conf.model_name):
    if id == -1:
        raise Exception("cannot continue wandb run. no valid  run id")
    global _wandb_run
    print("continuing wandb run, id=", id)
    _wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=get_config().cfg, resume=True, name=model_name, id=id)
    return _wandb_run
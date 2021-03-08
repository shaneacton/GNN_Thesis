from Config.config import conf

try:
    import wandb
    use_wandb = True
except Exception as e:
    print("wandb error:", e)
    use_wandb = False


wandb_run = None


def new_run(model_name=conf.model_name, config=conf):
    global wandb_run
    global use_wandb
    id = wandb.util.generate_id()
    try:
        wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=conf, resume=True, name=model_name, id=id)
    except Exception as e:
        print("cannot init wandb session. turning off wandb logging")
        print(e)
        use_wandb = False
        return None
    config.wandb_id = id
    config.cfg["wandb_id"] = id
    return wandb_run


def continue_run(id, model_name=conf.model_name):
    if id == -1:
        raise Exception("cannot continue wandb run. no valid  run id")
    global wandb_run
    print("continuing wandb run, id=", id)
    wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=conf, resume=True, name=model_name, id=id)
    return wandb_run
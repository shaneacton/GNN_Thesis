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
    id = wandb.util.generate_id()
    wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=conf, resume=True, name=model_name, id=id)
    config.wandb_id = id
    return wandb_run


def continue_run(id, model_name=conf.model_name):
    global wandb_run
    wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=conf, resume=True, name=model_name, id=id)
    return wandb_run
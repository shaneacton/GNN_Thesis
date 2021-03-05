try:
    import wandb
    use_wandb = True
except Exception as e:
    print("wandb error:", e)
    use_wandb = False

if use_wandb:
    wandb.init(project="gnn_thesis", entity="shaneacton")

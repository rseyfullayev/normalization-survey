import wandb

def init_wandb(model_name, norm_type, batch_size, epochs):
    wandb.init(
        project="normalization_techniques",
        config={
            "model": model_name,
            "normalization": norm_type,
            "batch_size": batch_size,
            "epochs": epochs
        }
    )


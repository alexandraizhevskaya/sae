import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pydantic import BaseModel
from tqdm.notebook import tqdm
from model import TopKSAE


def train_sae_model(train_loader_sae: DataLoader,
                    val_loader_sae: DataLoader,
                    train_loader: DataLoader,
                    train_sae: torch.utils.data.TensorDataset,
                    val_sae: torch.utils.data.TensorDataset,
                    training_params: BaseModel,
                    clip_model):

    # model
    sae = TopKSAE(input_d=training_params.input_d,
              latent_n=training_params.latent_n,
              top_k=training_params.top_k,
              lambda_=training_params.lambda_).to(training_params.device)
    sae.init_median_bias(vit=clip_model, dataloader = train_loader, latent_n=training_params.num_b)

    # optimization
    optimizer = torch.optim.AdamW(sae.parameters(),
                                lr=training_params.lr_sae,
                                weight_decay=training_params.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=training_params.epochs_sae)

    # loop
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(1, training_params.epochs_sae + 1)):

        # TRAIN
        sae.train()
        train_loss = 0.0
        for (x,) in tqdm(train_loader_sae):
            x = x.to(training_params.device, non_blocking=True)
            loss = sae.loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.renorm()

            train_loss += loss.item() * x.size(0)

        scheduler.step()

        # VAL
        sae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader_sae:
                x = x.to(training_params.device, non_blocking=True)
                loss = sae.loss(x)
                val_loss += sae.loss(x).item() * x.size(0)

        train_loss /= len(train_sae)
        val_loss /= len(val_sae)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # log
        if epoch % 5 == 0 or epoch == training_params.epochs_sae:
            print(f"Epoch {epoch} | train loss {train_loss:.5f} | val loss {val_loss:.5f}")

    torch.save(sae.state_dict(), training_params.save_file)
    print(f"Checkpoint saved to {training_params.save_file}")

    return sae, train_losses, val_losses
    
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import clip
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from IPython.display import display


def get_img_emdeddings(loader: DataLoader,
                       clip_model: clip.model.CLIP,
                       device: str = 'cuda'
                       ) -> torch.tensor:
  features = []
  clip_model.eval()

  with torch.no_grad():
      for img in tqdm(loader):
          z = clip_model.encode_image(img.to(device)).float()
          features.append(z.cpu())

  X_embed = torch.cat(features, dim=0)
  return X_embed


@torch.no_grad()
def get_active_neurons(
    z: torch.Tensor,
    th: float = 1.0,
    return_values: bool = False,
):
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
    if z.ndim == 1:
        mask = z.abs() > th
        idx  = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if return_values:
            vals = z[mask].tolist()
            return list(zip(idx, vals))
        return idx
    elif z.ndim == 2:
        out = []
        for row in z:
            mask = row.abs() > th
            out.append(torch.nonzero(mask, as_tuple=False).flatten().tolist())
        return out
    else:
        raise ValueError("Only 1-D or 2-D  tensors are supported")


def visualize_results(train_losses: list, val_losses: list) -> None:

    plt.figure(figsize=(25, 6))
    plt.plot(train_losses, label='train loss', color='orange')
    plt.plot(val_losses, label='val loss', color='blue')
    plt.legend()
    plt.xticks(ticks=range(len(train_losses)), labels=range(1, len(train_losses) + 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.title('Losses')
    plt.show()


def display_images_row(image_paths):
    images = [Image.open(path) for path in image_paths]
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))
    if len(images) == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_tsne(latents, km):
    reducer = TSNE(n_components=2)
    emb2d = reducer.fit_transform(latents)
    labels = km.predict(latents)
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(emb2d[:,0], emb2d[:,1], c=labels, s=5, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title='Cluster')
    plt.title('t-SNE of Latents')
    plt.tight_layout()
    plt.savefig('tsne.png')


def find_closest(kn, latents, idx, paths_val):
    dists, idxes = kn.kneighbors(latents[idx][None, :], n_neighbors=6)
    print("Query")
    display(Image.open(paths_val[idx]).resize((256, 256)))
    print()
    print("Closesest")
    img_cls_paths = [paths_val[i] for i in idxes[0][1:]]
    display_images_row(img_cls_paths)


def get_tokens(model, path, device):

    with open(path, 'r') as f:
      concepts = f.readlines()

    model.eval()
    concept_list = []
    embds = []
    with torch.no_grad():
      for concept in tqdm(concepts):
          concept = concept.strip('\n')
          tokens = clip.tokenize([concept]).to(device)
          emb = model.encode_text(tokens)
          concept_list.append(concept)
          embds.append(emb)
    emb = torch.cat(embds, dim=0)
    return np.array(concept_list), emb.cpu().numpy()


def save_npy(data, output_path):
    with open(output_path, 'wb') as f:
        np.save(f, data.to('cpu').numpy())


def load_npy(output_path):
    with open(output_path, 'rb') as f:
        data = np.load(f)
    return torch.tensor(data)

    
# https://github.com/WolodjaZ/MSAE/tree/main
def identify_dead_neurons(latent_bias: torch.Tensor, threshold: float = 10**(-5.5)) -> torch.Tensor:
    """
    Identify dead neurons based on their bias values.

    Dead neurons are those with bias magnitudes below a specified threshold,
    indicating that they may not be activating significantly during training.

    Args:
        latent_bias (torch.Tensor): Bias vector for latent neurons
        threshold (float, optional): Threshold below which a neuron is considered dead.
                                     Defaults to 10^(-5.5).

    Returns:
        torch.Tensor: Indices of dead neurons
    """
    dead_neurons = torch.where(torch.abs(latent_bias) < threshold)[0]
    return dead_neurons

def explained_variance_full(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    """
    Computes the explained variance between the original input and its reconstruction.

    The explained variance is a measure of how much of the variance in the original input
    is captured by the reconstruction. It is calculated as:
        1 - (variance of the reconstruction error / total variance of the original input)

    Args:
        original_input (torch.Tensor): The original input tensor.
        reconstruction (torch.Tensor): The reconstructed tensor.

    Returns:
        float: The explained variance score, a value between 0 and 1.
            A value of 1 indicates perfect reconstruction.
    """
    variance = (original_input - reconstruction).var(dim=-1)
    total_variance = original_input.var(dim=-1)
    return variance / total_variance


def normalized_mean_absolute_error(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute normalized mean absolute error between original and reconstructed data.

    This metric normalizes the MAE by the mean absolute value of the original input,
    making it scale-invariant and more comparable across different datasets.

    Args:
        original_input (torch.Tensor): Original input data of shape [batch, n_inputs]
        reconstruction (torch.Tensor): Reconstructed data of shape [batch, n_inputs]

    Returns:
        torch.Tensor: Normalized MAE for each sample in the batch
    """
    return (
        torch.abs(reconstruction - original_input).mean(dim=1) /
        torch.abs(original_input).mean(dim=1)
    )


def l0_messure(sample: torch.Tensor) -> torch.Tensor:
    """
    Compute the L0 measure (sparsity) of feature activations.

    The L0 measure counts the proportion of zero elements in the activation,
    providing a direct measure of sparsity. Higher values indicate greater
    sparsity (more zeros).

    Note: The function name contains a spelling variant ("messure" vs "measure")
    but is kept for backward compatibility.

    Args:
        sample (torch.Tensor): Activation tensor of shape [batch, n_features]

    Returns:
        torch.Tensor: Proportion of zero elements for each sample in the batch
    """
    return (sample == 0).float().mean(dim=1)


def score_representations(model, dataloader, device) -> dict:
    """
    Score representation by progressivly extracting TopK actviations

    Args:
        model: The Sparse Autoencoder model to evaluate
        dataset: Dataset to process
        batch_size (int): Number of samples to process at once

    Returns:
        dict with results

    Metrics computed:
        - Fraction of Variance Unexplained (FVU) using normalized MSE
        - Normalized Mean Absolute Error (MAE)
        - Cosine similarity between inputs and outputs
        - L0 measure (average number of active neurons per sample)
        - CKNNA (Cumulative k-Nearest Neighbor Accuracy)
        - Number of dead neurons (neurons that never activate)
    """

    model.eval()
    model.to(device)
    results = {}
    with torch.no_grad():

            # Lists to collect metrics for each batch
            l0 = []
            mae = []
            fvu = []
            cs = []
            cknnas = []
            dead_neurons_count = None

            # Process data in batches
            for idx, (batch,) in enumerate(tqdm(dataloader, desc="Extracting representations")):
                batch = batch.to(device)

                # Forward pass through the model
                with torch.no_grad():
                    _, representations, outputs = model(batch)

                # Post-process outputs and batch
                batch = batch.cpu()
                outputs = outputs.cpu()

                # Calculate and collect metrics
                fvu.append(explained_variance_full(batch, outputs))
                mae.append(normalized_mean_absolute_error(batch, outputs))
                cs.append(torch.nn.functional.cosine_similarity(batch, outputs))
                l0.append(l0_messure(representations))
                # Only calculate the cknna if it even to the number of the batch

                # Track neurons that are activated at least once
                if dead_neurons_count is None:
                    dead_neurons_count = (representations != 0).sum(dim=0).cpu().long()
                else:
                    dead_neurons_count += (representations != 0).sum(dim=0).cpu().long()

            # Aggregate metrics across all batches
            mae = torch.cat(mae, dim=0).cpu().numpy()
            cs = torch.cat(cs, dim=0).cpu().numpy()
            l0 = torch.cat(l0, dim=0).cpu().numpy()
            fvu = torch.cat(fvu, dim=0).cpu().numpy()

            # Count neurons that were never activated
            number_of_dead_neurons = torch.where(dead_neurons_count == 0)[0].shape[0]

            # Log final metrics
            k = model.top_k
            print(f"TopK: {k}")
            print(f"Fraction of Variance Unexplained (FVU): {np.mean(fvu)} +/- {np.std(fvu)}")
            print(f"Normalized MAE: {np.mean(mae)} +/- {np.std(mae)}")
            print(f"Cosine similarity: {np.mean(cs)} +/- {np.std(cs)}")
            print(f"L0 messure: {np.mean(l0)} +/- {np.std(l0)}")
            print(f"Number of dead neurons: {number_of_dead_neurons}")

            # Store results in dictionary
            results[k] = {
                "fvu": (float(np.mean(fvu)), float(np.std(fvu))),
                "mae": (float(np.mean(mae)), float(np.std(mae))),
                "cs": (float(np.mean(cs)), float(np.std(cs))),
                "l0": (float(np.mean(l0)), float(np.std(l0))),
                "number_of_dead_neurons": number_of_dead_neurons
            }

    # Return results
    return results

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image


class ClipImageDataset(Dataset):
    """class to preprocess images with clip preprocessor"""
    def __init__(self,
                 image_paths: list,
                 preprocess: torchvision.transforms.transforms.Compose,
            ) -> None:
      self.image_paths = image_paths
      self.preprocess = preprocess

    def __len__(self) -> int:
      return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.tensor:
      img = Image.open(self.image_paths[idx]).convert("RGB")
      img = self.preprocess(img)
      return img
      
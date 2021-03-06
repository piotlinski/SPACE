import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
import h5py

class MultiScaleMNIST(Dataset):

    def __init__(
        self,
        root: str,
        subset: str = "train",
    ):
        super().__init__()
        self.file_path = f"{root}/multiscalemnist.h5"
        self.subset = "train"
        with h5py.File(self.file_path, "r") as file:
            self.dataset_length = len(file[self.subset]["images"])
        self.dataset: Optional[h5py.File] = None

    def __len__(self):
        """Get dataset length."""
        return self.dataset_length

    def __getitem__(self, item):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.subset]
        image = self.dataset["images"][item]
        pil_image = Image.fromarray(image).resize((128, 128), Image.BICUBIC)
        image = np.array(pil_image) / 255
        return torch.from_numpy(image).float().permute(2, 0, 1)

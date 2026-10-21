from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)

        with gzip.open(image_filename, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(
                ">IIII", f.read(16)
            )

            self.images = np.frombuffer(
                f.read(), dtype=np.uint8
            ).reshape(num_images, rows, cols, 1)

            self.images = self.images.astype(np.float32) / 255.0

        with gzip.open(label_filename, "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))

            self.labels = np.frombuffer(
                f.read(), dtype=np.uint8
            ).copy()

    def __getitem__(self, index) -> object:
        img = self.images[index]
        label = self.labels[index]

        if img.ndim == 3:
            img = self.apply_transforms(img)
        elif self.transforms is not None:
            img = np.stack([self.apply_transforms(image) for image in img], axis=0)

        return img, label

    def __len__(self) -> int:
        return len(self.images)

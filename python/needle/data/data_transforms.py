import numpy as np
from abc import ABC, abstractmethod

class Transform(ABC):
    @abstractmethod
    def __call__(self, x):
        """Apply a transform to an input."""


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
            img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        H, W, C = img.shape
        p = self.padding

        padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode='constant', constant_values=0)

        return padded[p + shift_x : p + shift_x + H, p + shift_y : p + shift_y + W,:]

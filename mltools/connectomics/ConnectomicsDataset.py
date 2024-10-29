import torch
import glob
import os
import numpy as np
from PIL import Image
import scipy.ndimage as sim
import skimage.morphology as skmorph
import h5py
import tqdm
import cv2

clahe = cv2.createCLAHE(clipLimit=3).apply


class PatchAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, n_samples, patch_size=256):
        super().__init__()
        self.h5 = h5py.File(file_path, "r")
        self.n_samples = n_samples  # does not affect actual memory
        if "W" not in self.h5.attrs.keys() or "H" not in self.h5.attrs.keys():
            raise ValueError(
                "W and H must be in the attributes of the h5 file, different sizes not implemented"
            )
        self.W = self.h5.attrs["W"]
        self.H = self.h5.attrs["H"]
        self.C = self.h5.attrs["C"]

        self.N_images = self.h5.attrs["T"]

        self.im_dtype = None
        self.mask_dtype = None
        self.ims_masks = {}
        with tqdm.tqdm(total=self.N_images) as pbar:
            for i in range(self.N_images):
                im, mask = (
                    np.array(self.h5[str(i) + "/im"])[0, :, :, 0],
                    np.array(self.h5[str(i) + "/mask"])[0, :, :, 0],
                )
                im_clahe = clahe(im)
                self.ims_masks[str(i)] = im, im_clahe, mask
                pbar.update(1)
        self.im_dtype = im.dtype
        self.mask_dtype = mask.dtype
        self.h5.close()
        self.patch_size = patch_size

        self.grid = (
            np.stack(
                np.meshgrid(
                    np.arange(self.patch_size),
                    np.arange(self.patch_size),
                    indexing="ij",
                ),
                axis=0,
            )
            - self.patch_size / 2
            + 0.5
        )
        self.out = int(np.sqrt(2) * (self.patch_size // 2 + 1) + 1)

        self.count = np.zeros(self.N_images)
        self.mincount = 0

    def get_random_image_mask(self, p_clahe=0.5):
        candidates = self.count == self.mincount
        if candidates.sum() == 0:
            self.mincount += 1
            candidates = self.count == self.mincount
        i = np.random.choice(np.nonzero(candidates)[0])
        self.count[i] += 1
        im, im_clahe, mask = self.ims_masks[str(i)]
        if np.random.random() < p_clahe:
            return im_clahe, mask
        return im, mask

    def __getitem__(self, i):
        if (not isinstance(i, int)) or i < 0 or i >= self.n_samples:
            raise IndexError
        loc = (
            self.out
            + np.array(
                [
                    np.random.choice(self.W - 2 * self.out),
                    np.random.choice(self.H - 2 * self.out),
                ]
            )
            + np.random.random()
            - 0.5
        )
        theta = np.random.random() * 2 * np.pi
        rotmat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        grid_ = np.einsum("ij,jkm->ikm", rotmat, self.grid)
        if np.random.random() < 0.5:
            grid_[0] *= -1
        grid_ += loc[:, None, None]

        im, mask = self.get_random_image_mask()
        im_ = sim.map_coordinates(im, [grid_[0], grid_[1]], order=0)
        mask_ = sim.map_coordinates(mask, [grid_[0], grid_[1]], order=0)
        return torch.from_numpy(im_ / np.iinfo(self.im_dtype).max)[None].to(
            dtype=torch.float32
        ), torch.from_numpy(mask_ / np.iinfo(self.mask_dtype).max).to(dtype=torch.int64)

    def __len__(self):
        return self.n_samples

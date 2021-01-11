from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from libtiff import TIFF
import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', tif=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.tif = tif
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, cv2_img, scale):
        w = cv2_img.shape[0]
        h = cv2_img.shape[1]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img_nd = np.array(cv2_img)

        if len(cv2_img.shape) > 2:

            img_nd = img_nd.transpose((2, 0, 1))
            if img_nd.max() > 1:
                img_nd = img_nd / 255

        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = TIFF.open(mask_file[0], mode='r').read_image()
        img = TIFF.open(img_file[0], mode='r').read_image()

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # print(mask//80)

        return {
            'image': torch.from_numpy(img).type(torch.float32),
            'mask': torch.from_numpy(mask//80).type(torch.uint8)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, tif=False):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask', tif=tif)

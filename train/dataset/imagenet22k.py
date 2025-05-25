import os
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
from lib.train.dataset.imagenet22k_dataset import IN22KDataset
from collections import OrderedDict
from lib.train.admin import env_settings
import numpy as np


class Imagenet22k(BaseVideoDataset):
    """ The ImageNet22k dataset. ImageNet22k is an image dataset. Thus, we treat each image as a sequence of length 1.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split="train"):
        """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
        """
        root = env_settings().imagenet22k_dir if root is None else root
        super().__init__('imagenet22k', root, image_loader)

        self.dataset = IN22KDataset(data_root=root, transform=None, fname_format='imagenet5k/{}.JPEG', debug=False)

    def is_video_sequence(self):
        return False

    def get_name(self):
        return 'imagenet22k'

    def get_num_sequences(self):
        return len(self.dataset)

    def get_sequence_info(self, seq_id):
        '''2021.1.3 To avoid too small bounding boxes. Here we change the threshold to 50 pixels'''
        valid = torch.tensor([True])
        visible = valid.clone().byte()
        return {'bbox': None, 'mask': None, 'valid': valid, 'visible': visible}

    def _get_frames(self, seq_id):
        img, target = self.dataset.__getitem__(seq_id)
        return img

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # Imagenet is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        return frame_list, None, None


if __name__ == '__main__':
    data_root = './data/imagenet22k'
    dataset = Imagenet22k(data_root)

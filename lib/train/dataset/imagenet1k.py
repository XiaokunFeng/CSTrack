import os
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
import torchvision.datasets as datasets
from collections import OrderedDict
from lib.train.admin import env_settings


class Imagenet1k(BaseVideoDataset):
    """ The ImageNet1k dataset. ImageNet1k is an image dataset. Thus, we treat each image as a sequence of length 1.
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
        root = env_settings().imagenet1k_dir if root is None else root
        super().__init__('imagenet1k', root, image_loader)

        self.dataset = datasets.ImageFolder(os.path.join(root, 'train'), loader=image_loader)

    def is_video_sequence(self):
        return False

    def get_name(self):
        return 'imagenet1k'

    def get_num_sequences(self):
        return len(self.dataset.samples)

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
    data_root = './data/imagenet1k'
    dataset = Imagenet1k(data_root)
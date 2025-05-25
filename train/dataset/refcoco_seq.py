import os
import random
from collections import OrderedDict

import torch
import numpy as np

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from .refer import REFER


class RefCOCOSeq(BaseVideoDataset):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
                - instances_train2017.json
            - images
                - train2014
                - train2017

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2014",
                 name="refcoco", splitBy="google", multi_modal_vision=False, multi_modal_language=False):
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
        root = env_settings().coco_dir if root is None else root
        super().__init__('RefCOCOSeq', root, image_loader)
        self.split = split
        self.img_pth = os.path.join(root, 'images/{}{}'.format("train", version))
        self.anno_path = os.path.join(root, '{}/instances.json'.format(name))
        self.dataset_name = name
        # Load the COCO set.
        self.coco_set = REFER(root, dataset=name, splitBy=splitBy)

        self.cats = self.coco_set.Cats

        self.class_list = self.get_class_list()

        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        self.seq_per_class = self._build_seq_per_class()

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def _get_sequence_list(self):
        ref_list = list(self.coco_set.getRefIds(split=self.split))
        seq_list = [a for a in ref_list if self.coco_set.refToAnn[a]['iscrowd'] == 0]

        return seq_list

    def is_video_sequence(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return self.dataset_name

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id])
        return class_list

    def has_segmentation_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = self.cats[self.coco_set.refToAnn[seq]['category_id']]
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)

        bbox = torch.Tensor(anno['bbox']).view(1, 4)

        # mask = torch.Tensor(self.coco_set.annToMask(anno)).unsqueeze(dim=0)

        '''2021.1.3 To avoid too small bounding boxes. Here we change the threshold to 50 pixels'''
        valid = (bbox[:, 2] > 50) & (bbox[:, 3] > 50)

        visible = valid.clone().byte()

        output = {'bbox': bbox, 'valid': valid, 'visible': visible}
        if self.multi_modal_language:
            nlp = self._read_nlp(seq_id)
            output['nlp'] = nlp

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'nlp': nlp}

    def _read_nlp(self, seq_id):

        ref = self.coco_set.Refs[self.sequence_list[seq_id]]
        sent = ref['sentences'][-1]['sent']

        return sent

    def _get_anno(self, seq_id):
        anno = self.coco_set.refToAnn[self.sequence_list[seq_id]]

        return anno

    def _get_frames(self, seq_id):
        path = self.coco_set.loadImgs([self.coco_set.refToAnn[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        if self.multi_modal_vision:
            img = np.concatenate((img, img), axis=-1)
        return img

    def get_meta_info(self, seq_id):
        try:
            cat_dict_current = self.cats[self.coco_set.refToAnn[self.sequence_list[seq_id]]['category_id']]
            object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def get_class_name(self, seq_id):
        cat_dict_current = self.cats[self.coco_set.refToAnn[self.sequence_list[seq_id]]['category_id']]
        return cat_dict_current['name']

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta

    def get_path(self, seq_id, frame_ids):
        img_name = self.coco_set.loadImgs([self.coco_set.refToAnn[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        return [os.path.join(self.img_pth, img_name) for _ in range(len(frame_ids))]

    def get_ref_id(self, seq_id):
        return self.sequence_list[seq_id]

import os
import os.path
import random
import re
from collections import OrderedDict

#import ipdb
import numpy as np
import pandas
import torch

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset


class Tnl2k(BaseVideoDataset):
    """ TNL2K dataset.

    Publication:
        Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark
        Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu CVPR 2021 2021

    Download the dataset from https://sites.google.com/view/langtrackbenchmark/
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=False):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().tnl2k_dir if root is None else root
        super().__init__('Tnl2k', root, image_loader)
        # Keep a list of all sequence path
        self.sequence_path = [f for f in os.listdir(self.root)]
        self.block_list = ['CM', 'INF']
        self.sequence_list = self._build_sequence_list(vid_ids, split)
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def _build_sequence_list(self, vid_ids=None, split=None):
        # todo update split
        sequence_list = []
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'tnl2k_train_split.txt')
            else:
                raise ValueError('Unknown split name.')
            # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c + '-' + str(v) for c in self.class_list for v in vid_ids]
        elif vid_ids is None and split is None:
            sequence_list = self.sequence_path
            # raise ValueError('Set either split_name or vid_ids.')
        return sequence_list

    def _build_class_list(self):
        # if not self.has_class_info():
        return None

    def get_name(self):
        return 'tnl2k'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return None

    def get_sequences_in_class(self, class_name):
        if self.has_class_info():
            return self.seq_per_class[class_name]
        else:
            return None

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_nlp(self, seq_path):
        nlp_file = os.path.join(seq_path, "language.txt")
        nlp = ""
        try:
            nlp = pandas.read_csv(nlp_file, dtype=str, header=None, low_memory=False).values
        except Exception as e:
            print(e)
            print(f'nlp_file:{nlp_file}')
        return nlp[0][0]

    def _read_target_visible(self, seq_path):
        # Read groundtruth.txt
        bbox = self._read_bb_anno(seq_path)

        target_visible = (bbox[:, 0] > 0) | (bbox[:, 1] > 0) | (bbox[:, 2] > 0) | (bbox[:, 3] > 0)

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]

        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path).bool() & valid.byte().bool()
        output = {'bbox': bbox, 'valid': valid, 'visible': visible}
        # if self.multi_modal_language:
        #     nlp = self._read_nlp(seq_path)
        #     output['nlp'] = nlp
        return output

    def get_sequence_nlp(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        nlp = self._read_nlp(seq_path)
        return nlp

    def _get_frame_path(self, seq_path, frame_id):
        images_path = sorted(os.listdir(os.path.join(seq_path, 'imgs')))
        image_path = os.path.join(seq_path, 'imgs', images_path[frame_id])
        return image_path

    def _get_frame(self, seq_path, frame_id):
        frame = self.image_loader(self._get_frame_path(seq_path, frame_id))
        if self.multi_modal_vision:
            frame = np.concatenate((frame, frame), axis=-1)
        return frame

    def _get_class(self, seq_path):
        seq_name = seq_path.split('/')[-1]
        raw_class = seq_name.split('_')[0]
        if raw_class in self.block_list:
            raw_class = seq_name.split('_')[1]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_path(self, seq_id, frame_ids):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame_path(seq_path, f_id) for f_id in frame_ids]
        return frame_list

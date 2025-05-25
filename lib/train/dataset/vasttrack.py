import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import re
# from lib.utils.string_utils import clean_string
from lib.train.dataset.vasttrack_test.utils import load_text, Sequence, SequenceList

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')

class VastTrack(BaseVideoDataset):
    """ VastTrack dataset.

    Publication:
        VastTrack: Vast Category Visual Object Tracking
        Liang Peng, Junyuan Gao, Xinran Liu, Weihong Li, Shaohua Dong, Zhipeng Zhang, Heng Fan and Libo Zhang
        https://arxiv.org/pdf/2403.03493.pdf

    Download the dataset from https://github.com/HengLan/VastTrack
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
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
        root = env_settings().vasttrack_dir if root is None else root
        super().__init__('Vasttrack', root, image_loader)

        # get test split list infor
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        self.test_seq_list_path = os.path.join(ltr_path, "vasttrack_test/vasttrack_test_list.txt")

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)]
        # debug_filter  = []
        # for item in self.class_list:
        #     if "Ru" in item:
        #         debug_filter.append(item)
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        # if split is not None:
        #     if vid_ids is not None:
        #         raise ValueError('Cannot set both split_name and vid_ids.')
        #     ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        #     if split == 'train':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
        #     else:
        #         raise ValueError('Unknown split name.')
        #     sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        # elif vid_ids is not None:
        #     sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        # else:
        #     raise ValueError('Set either split_name or vid_ids.')

        ## todo: split train and test list ; this version uses all data for train
        sequence_list = [ ]
        for c in self.class_list:
            vid_ids_item  = os.listdir(os.path.join(self.root,c))
            sequence_list += vid_ids_item

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'vasttrack_test'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "Groundtruth.txt")
        # ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        # occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        # out_of_view_file = os.path.join(seq_path, "out_of_view.txt")
        #
        # with open(occlusion_file, 'r', newline='') as f:
        #     occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        # with open(out_of_view_file, 'r') as f:
        #     out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        #
        # target_visible = ~occlusion & ~out_of_view
        bb_anno_file = os.path.join(seq_path, "Groundtruth.txt")
        ground_truth_rect = load_text(str(bb_anno_file), delimiter=',', dtype=np.float64)
        target_visible = ~((ground_truth_rect == [0, 0, 0, 0]).all(axis=1))
        target_visible = torch.tensor(target_visible)
        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        vid_id = seq_name.split('-')[-1]
        class_name = seq_name[:-1*(len(vid_id)+1)]

        return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # visible = self._read_target_visible(seq_path) & valid.byte()
        visible = self._read_target_visible(seq_path)
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'imgs', '{:05}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def _get_expression(self, seq_path):
        # read expression data
        exp_path = os.path.join(seq_path, 'nlp.txt')
        exp_str = ''
        try:
            with open(exp_path, 'r') as f:
                for line in f.readlines():
                    exp_str += line
        except Exception as e:
            print(e)
            # return None

        assert (exp_str != '' and not exp_str is None), 'ERROR: Language File is None: "{}"'.format(exp_path)
        exp_str = clean_string(exp_str)
        return exp_str

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        exp_str = self._get_expression(seq_path) # read expression data
        
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'exp_str': exp_str})

        return frame_list, anno_frames, object_meta

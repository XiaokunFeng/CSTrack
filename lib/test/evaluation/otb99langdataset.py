import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class OTB99LangDataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otblang_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        sequence_name = sequence_info['name']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/OTB_videos/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/OTB_videos/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        nlp_path = '{}/{}/{}'.format(self.base_path, "OTB_query_test", "{}.txt".format(sequence_name))
        nlp_rect = load_text(str(nlp_path), delimiter=',', dtype=str)
        nlp_rect = str(nlp_rect)

        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'], language_query=nlp_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg",
             "anno_path": "Biker/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Bird1", "path": "Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg",
             "anno_path": "Bird1/groundtruth_rect.txt",
             "object_class": "bird"},
            {"name": "Bird2", "path": "Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg",
             "anno_path": "Bird2/groundtruth_rect.txt",
             "object_class": "bird"},
            {"name": "BlurBody", "path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg",
             "anno_path": "BlurBody/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "BlurCar1", "path": "BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar1/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar2", "path": "BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar2/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar3", "path": "BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar3/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar4", "path": "BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar4/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurFace", "path": "BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg",
             "anno_path": "BlurFace/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "BlurOwl", "path": "BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg",
             "anno_path": "BlurOwl/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg",
             "anno_path": "Board/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Bolt2", "path": "Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg",
             "anno_path": "Bolt2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Box", "path": "Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg",
             "anno_path": "Box/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Car1", "path": "Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg",
             "anno_path": "Car1/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Car2", "path": "Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg",
             "anno_path": "Car2/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Car24", "path": "Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg",
             "anno_path": "Car24/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Coupon", "path": "Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg",
             "anno_path": "Coupon/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Crowds", "path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg",
             "anno_path": "Crowds/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dancer", "path": "Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg",
             "anno_path": "Dancer/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dancer2", "path": "Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg",
             "anno_path": "Dancer2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg",
             "anno_path": "Diving/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dog", "path": "Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg",
             "anno_path": "Dog/groundtruth_rect.txt",
             "object_class": "dog"},
            {"name": "DragonBaby", "path": "DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg",
             "anno_path": "DragonBaby/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
             "anno_path": "Girl2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg",
             "anno_path": "Gym/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human2", "path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg",
             "anno_path": "Human2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human3", "path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg",
             "anno_path": "Human3/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human4_2", "path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg",
             "anno_path": "Human4/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Human5", "path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg",
             "anno_path": "Human5/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human6", "path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg",
             "anno_path": "Human6/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human7", "path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg",
             "anno_path": "Human7/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human8", "path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg",
             "anno_path": "Human8/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human9", "path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg",
             "anno_path": "Human9/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Jump", "path": "Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg",
             "anno_path": "Jump/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "KiteSurf", "path": "KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": "jpg",
             "anno_path": "KiteSurf/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Man", "path": "Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg",
             "anno_path": "Man/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg",
             "anno_path": "Panda/groundtruth_rect.txt",
             "object_class": "mammal"},
            {"name": "RedTeam", "path": "RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg",
             "anno_path": "RedTeam/groundtruth_rect.txt",
             "object_class": "vehicle"},
            {"name": "Rubik", "path": "Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg",
             "anno_path": "Rubik/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Skater", "path": "Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg",
             "anno_path": "Skater/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skater2", "path": "Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg",
             "anno_path": "Skater2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skating2-1", "path": "Skating2-1/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
             "anno_path": "Skating2-1/groundtruth_rect.1.txt",
             "object_class": "person"},
            {"name": "Skating2-2", "path": "Skating2-2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
             "anno_path": "Skating2-2/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Surfer", "path": "Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg",
             "anno_path": "Surfer/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Toy", "path": "Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
             "anno_path": "Toy/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Trans", "path": "Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg",
             "anno_path": "Trans/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Twinnings", "path": "Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg",
             "anno_path": "Twinnings/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Vase", "path": "Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
             "anno_path": "Vase/groundtruth_rect.txt",
             "object_class": "other"},
        ]

        return sequence_info_list
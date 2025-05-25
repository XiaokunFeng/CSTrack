import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
import importlib

prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

prj = join(dirname(__file__), '../..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.train.dataset.depth_utils import get_x_frame
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
from lib.test.evaluation.environment import env_settings


def genConfig(seq_path, set_type):
    if set_type == 'VisEvent':
        RGB_img_list = sorted([seq_path + '/vis_imgs/' + p for p in os.listdir(seq_path + '/vis_imgs') if os.path.splitext(p)[1] == '.bmp'])
        E_img_list = sorted([seq_path + '/event_imgs/' + p for p in os.listdir(seq_path + '/event_imgs') if os.path.splitext(p)[1] == '.bmp'])

        RGB_gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        absent_label = np.loadtxt(seq_path + '/absent_label.txt')

    return RGB_img_list, E_img_list, RGB_gt, absent_label

def get_parameters(script_name, yaml_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(script_name))
    params = param_module.parameters(yaml_name)
    return params

def create_tracker(script_name, params, dataset_name):
    tracker_module = importlib.import_module('lib.test.tracker.{}'.format(script_name))
    tracker_class = tracker_module.get_tracker_class()
    tracker = tracker_class(params, dataset_name)
    return tracker

def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, checkpoint_name="", debug=0, script_name='prompt'):
    seq_txt = seq_name
    epoch_idex = checkpoint_name.split(".")[0]
    epoch_idex = epoch_idex.split("_")[-1]
    # save_name = '{}_ep{}'.format(yaml_name, epoch)
    save_name = '{}'.format(yaml_name)
    save_path = f'{env_settings().prj_dir}/output/test/tracking_results/{save_name}/' + dataset_name + "_" + epoch_idex + '/' + seq_txt + '.txt'
    save_folder = f'{env_settings().prj_dir}/output/test/tracking_results/{save_name}/' + dataset_name + "_" + epoch_idex
    try:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    except:
        print("exist")
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    params = get_parameters(script_name, yaml_name)
    params.checkpoint = os.path.join(env_settings().network_path, checkpoint_name)
    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    params.debug = debug_
    mmtrack = create_tracker(script_name, params, 'visevent')
    tracker = SEQTRACK_RGBE(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+ seq_name +'——————————————')
    RGB_img_list, E_img_list, RGB_gt, absent_label = genConfig(seq_path, dataset_name)
    if absent_label[0] == 0: # first frame is absent in some seqs
        first_present_idx = absent_label.argmax()
        RGB_img_list = RGB_img_list[first_present_idx:]
        E_img_list = E_img_list[first_present_idx:]
        RGB_gt = RGB_gt[first_present_idx:]
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    # try:
    for frame_idx, (rgb_path, E_path) in enumerate(zip(RGB_img_list, E_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            region, confidence = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class SEQTRACK_RGBE(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # ostrack_ce_ep60_adapter_iv21b_wofovea_8_onlylasher_2xa100_rgbt
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='cstrack_s2',
                        help='Name of tracking method(ostrack, adapter, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='cstrack_s2',
                        help='Name of tracking method.')  # vitb_256_mae_ce_32x4_ep300 vitb_256_mae_ce_32x4_ep60_adapter_i32v21_onlylasher_rgbt
    parser.add_argument('--dataset_name', type=str, default='VisEvent',
                        help='Name of dataset (VisEvent).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus')  # torch.cuda.device_count()
    parser.add_argument('--ckpt_name', type=str, default="SEQTRACKV2_ep0030.pth.tar")

    #################################-------------------------------##################
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))
    # path initialization
    seq_list = None
    if dataset_name == 'VisEvent':
        seq_home = env_settings().visevent_dir
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    checkpoint_name_list = [args.ckpt_name]

    for checkpoint_name in checkpoint_name_list:

        checkpoint_path = os.path.join(env_settings().network_path, checkpoint_name)
        # checkpoint_path = os.path.join(network_path, checkpoint_name)
        if os.path.isfile(checkpoint_path) == False:
            print("Can't load ", checkpoint_path)
            continue

        start = time.time()
        if args.mode == 'parallel':
            sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, checkpoint_name, args.debug,
                              args.script_name) for s in seq_list]
            multiprocessing.set_start_method('spawn', force=True)
            with multiprocessing.Pool(processes=args.threads) as pool:
                pool.starmap(run_sequence, sequence_list)
        else:
            seq_list = [args.video] if args.video != '' else seq_list
            sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, checkpoint_name, args.debug,
                              args.script_name) for s in seq_list]
            for seqlist in sequence_list:
                run_sequence(*seqlist)
        print(f"Totally cost {time.time() - start} seconds!")



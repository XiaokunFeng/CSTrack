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

import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
from lib.test.evaluation.environment import env_settings


def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')

    return RGB_img_list, T_img_list, RGB_gt, T_gt

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

def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1,checkpoint_name="", debug=0, script_name='prompt'):
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

    # if os.path.exists(save_path):
    #     print(f'-1 {seq_name}')
    #     return


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
    mmtrack = create_tracker(script_name, params, 'lasher')  # "GTOT" # dataset_name
    tracker = SEQTRACK_RGBT(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            region, confidence = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class SEQTRACK_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)

        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='cstrack_s2', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='cstrack_s2', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='RGBT234', help='Name of dataset (RGBT234,LasHeR).')
    parser.add_argument('--ckpt_name', type=str, default="SEQTRACKV2_ep0030.pth.tar")
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # path initialization
    seq_list = None
    if dataset_name == 'RGBT234':
        # seq_home = '/mnt/first/fengxiaokun_temp/video_ds/rgbt_ds/RGBT234'
        seq_home = env_settings().rgbt234_dir
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = env_settings().lasher_dir
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")
    checkpoint_name_list = [args.ckpt_name]

    for checkpoint_name in checkpoint_name_list:

        checkpoint_path = os.path.join(env_settings().network_path, checkpoint_name)
        # checkpoint_path = os.path.join(network_path, checkpoint_name)
        print("checkpoint_path",checkpoint_path)
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




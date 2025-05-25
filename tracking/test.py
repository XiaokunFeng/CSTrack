import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker, trackerlist
from lib.test.evaluation.environment import env_settings

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    # dataset = dataset[:2]
    if sequence is not None:
        dataset = [dataset[sequence]]


    eval_model_list = ["CSTRACK_S1_ep0180.pth.tar"
                       ]
    env = env_settings()
    try:
        ckpt_save_dir = env.checkpoints_path
    except:
        checkpoints_path = env.checkpoints_path
        if checkpoints_path.endswith('.pth.tar'):
            ckpt_save_dir = os.path.dirname(checkpoints_path)
        else:
            ckpt_save_dir = checkpoints_path
    results_dir = os.path.join(env.save_dir, tracker_param)
    if os.path.exists(results_dir) ==False:
        os.makedirs(results_dir)
    # trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    for model_item in eval_model_list:
        trackers = trackerlist(name=tracker_name, parameter_name=tracker_param, dataset_name=dataset_name,
                               run_ids=run_id)

        checkpoints_path_item = os.path.join(ckpt_save_dir, model_item)

        if os.path.isfile(checkpoints_path_item) == False:
            print("Don't exist :", checkpoints_path_item)
            continue
        epoch_idex = model_item.split(".")[0]
        epoch_idex=epoch_idex.split("_")[-1]

        results_dir_item = os.path.join(results_dir,dataset_name+"_"+epoch_idex+ "_st")

        trackers[0].results_eval_dir = results_dir_item
        trackers[0].results_dir = results_dir_item
        trackers[0].checkpoints_path = checkpoints_path_item

        print("checkpoints_path_item:", checkpoints_path_item)
        print("results_dir_item:", results_dir_item)
        run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default="cstrack_s1", help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default="cstrack_s1", help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default="lasot", help='Name of dataset.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)


    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()

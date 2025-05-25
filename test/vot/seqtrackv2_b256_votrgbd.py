import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.seqtrack_class import run_vot_exp

run_vot_exp('seqtrackv2', 'qt_v11_2_later_fusion_based_on_v4_9_all_finetune', vis=False, out_conf=True, channel_type='rgbd')

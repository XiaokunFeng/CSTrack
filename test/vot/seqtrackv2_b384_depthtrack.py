import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.seqtrack_class import run_vot_exp

run_vot_exp('seqtrackv2', 'seqtrackv2_b384', vis=False, out_conf=True, channel_type='rgbd')


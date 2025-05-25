class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/xkfeng/Python_proj/seqtrackv2'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/xkfeng/Python_proj/seqtrackv2/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/xkfeng/Python_proj/seqtrackv2/resource'

        # rgb
        self.got10k_dir = '/mnt/first/hushiyu/SOT/GOT-10k/data/train'
        self.got10k_val_dir = '/mnt/first/hushiyu/SOT/GOT-10k/data/val'
        self.got10k_lmdb_dir = '/mnt/first/hushiyu/SOT/GOT-10k/data/train'

        self.lasot_dir = '/mnt/first/hushiyu/SOT/LaSOT/data'
        self.lasot_lmdb_dir = '/mnt/first/hushiyu/SOT/LaSOT/data'

        self.trackingnet_dir = '/mnt/first/hushiyu/SOT/TrackingNet'
        self.trackingnet_lmdb_dir = '/mnt/first/hushiyu/SOT/TrackingNet'

        self.coco_lmdb_dir = '/mnt/first/hushiyu/SOT/COCO2017'
        self.coco_dir = '/mnt/first/hushiyu/SOT/COCO2017'

        self.tnl2k_dir = '/mnt/first/hushiyu/SOT/TNL2k/TNL2K_train_subset'
        self.vasttrack_dir = '/mnt/first/hushiyu/SOT/VastTrack/unisot_train_final_backup'  # for 28


        # rgbt
        self.lasher_dir = '/mnt/first/fengxiaokun_temp/video_ds/rgbt_ds/LasHeR0428/LasHeR_Divided_data/trainingset'

        # rgbe
        self.visevent_dir = '/mnt/first/fengxiaokun_temp/video_ds/rgbe_ds/VisEvent_dataset/train_subset'
        # rgbt
        self.depthtrack_dir = '/mnt/first/fengxiaokun_temp/video_ds/rgbd_ds/depthtrack_train/'

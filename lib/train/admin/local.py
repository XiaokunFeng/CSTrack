class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/path/to/your/project'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/path/to/your/project/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/path/to/your/project/resource'

        # rgb
        self.got10k_dir = '/path/to/your/data/GOT-10k/data/train'
        self.got10k_val_dir = '/path/to/your/data/GOT-10k/data/val'
        self.got10k_lmdb_dir = '/path/to/your/data/GOT-10k/data/train'

        self.lasot_dir = '/path/to/your/data/LaSOT/data'
        self.lasot_lmdb_dir = '/path/to/your/data/LaSOT/data'

        self.trackingnet_dir = '/path/to/your/data/TrackingNet'
        self.trackingnet_lmdb_dir = '/path/to/your/data/TrackingNet'

        self.coco_lmdb_dir = '/path/to/your/data/COCO2017'
        self.coco_dir = '/path/to/your/data/COCO2017'

        self.tnl2k_dir = '/path/to/your/data/TNL2k/TNL2K_train_subset'
        self.vasttrack_dir = '/path/to/your/data/VastTrack/unisot_train_final_backup'  # for 28


        # rgbt
        self.lasher_dir = '/path/to/your/data/LasHeR/LasHeR_Divided_data/trainingset'

        # rgbe
        self.visevent_dir = '/path/to/your/data/VisEvent_dataset/train_subset'
        # rgbt
        self.depthtrack_dir = '/path/to/your/data/depthtrack_train/'

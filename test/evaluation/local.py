from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    # settings.checkpoints_path = '/mnt/first/fengxiaokun_temp/model_checkpoint_save/seqtrackv2/checkpoints/train/seqtrackv2/qt_v4_10_involve_learning_prompts'   # Where tracking networks are stored.
    settings.checkpoints_path = '/mnt/first/fengxiaokun_temp/model_checkpoint_save/seqtrackv2/checkpoints/train/seqtrackv2/qt_v11_5_adjust_topk_tdlayer_1_adjust'  # Where tracking networks are stored.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/xkfeng/Python_proj/CSTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/xkfeng/Python_proj/CSTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/xkfeng/Python_proj/CSTrack/data/itb'
    settings.lasot_extension_subset_path = '/home/xkfeng/Python_proj/CSTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/xkfeng/Python_proj/CSTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/data/wargame/fengxiaokun/LaSOT/data'


    settings.network_path = settings.checkpoints_path    # Where tracking networks are stored.
    settings.nfs_path = '/home/xkfeng/Python_proj/CSTrack/data/nfs'
    settings.otb_lang_path = '/home/xkfeng/Python_proj/CSTrack/data/otb_lang'
    settings.otb_path = '/home/xkfeng/Python_proj/CSTrack/data/otb'
    settings.prj_dir = '/home/xkfeng/Python_proj/CSTrack'

    settings.result_plot_path = '/home/xkfeng/Python_proj/CSTrack/output/test/result_plots'
    settings.results_path = '/home/xkfeng/Python_proj/CSTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xkfeng/Python_proj/CSTrack/output'
    settings.segmentation_path = '/home/xkfeng/Python_proj/CSTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/xkfeng/Python_proj/CSTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/xkfeng/Python_proj/CSTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/xkfeng/Python_proj/CSTrack/data/trackingnet'

    # rgbt
    settings.lasher_dir = '/mnt/first/fengxiaokun_temp/video_ds/rgbt_ds/LasHeR0428/LasHeR_Divided_data/testingset'
    settings.rgbt234_dir = '/mnt/first/fengxiaokun_temp/video_ds/rgbt_ds/RGBT234'

    # rgbe
    settings.visevent_dir = '/mnt/first/fengxiaokun_temp/video_ds/rgbe_ds/VisEvent_dataset/test_subset'
    # rgbt
    settings.depthtrack_dir = '/home/data_3/video_ds/rgbd_ds/depthtrack_test/'

    return settings


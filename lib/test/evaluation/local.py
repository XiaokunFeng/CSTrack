from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.checkpoints_path = '/path/to/ckpt/save/dir'  # Where tracking networks are stored.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/path/to/your/project/data/got10k_lmdb'
    settings.got10k_path = '/path/to/your/project/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/path/to/your/project/data/itb'
    settings.lasot_extension_subset_path = '/path/to/your/project/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/path/to/your/project/data/lasot_lmdb'
    settings.lasot_path = '/path/to/data/LaSOT/data'


    settings.network_path = settings.checkpoints_path    # Where tracking networks are stored.
    settings.nfs_path = '/path/to/your/project/data/nfs'
    settings.otb_lang_path = '/path/to/your/project/data/otb_lang'
    settings.otb_path = '/path/to/your/project/data/otb'
    settings.prj_dir = '/path/to/your/project'

    settings.result_plot_path = '/path/to/your/project/output/test/result_plots'
    settings.results_path = '/path/to/your/project/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/path/to/your/project/output'
    settings.segmentation_path = '/path/to/your/project/output/test/segmentation_results'
    settings.tc128_path = '/path/to/your/project/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/path/to/your/project/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/path/to/your/project/data/trackingnet'

    # rgbt
    settings.lasher_dir = '/path/to/your/data/LasHeR/LasHeR_Divided_data/testingset'
    settings.rgbt234_dir = '/path/to/your/data/RGBT234'

    # rgbe
    settings.visevent_dir = '/path/to/your/data/VisEvent_dataset/test_subset'
    # rgbt
    settings.depthtrack_dir = '/path/to/your/data/depthtrack_test/'

    return settings


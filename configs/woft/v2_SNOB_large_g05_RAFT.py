from pathlib import Path
from flatsam.config import Config
from flatsam.woft_stuff.raft import RAFTWrapper
from flatsam import repo_path


def get_config():
    conf = Config()

    conf.of_class = RAFTWrapper
    conf.raft_type = 'weighted'

    conf.class_params = Config()
    conf.class_params.small = False
    conf.class_params.mixed_precision = False
    conf.class_params.alternate_corr = False
    conf.class_params.weight_head_structure = [(128, 3), (128, 3), (128, 3)]

    # start of defaults
    conf.class_params.mask_estimation = False
    conf.backbone_model = False
    # endo of defaults

    weight_dir = repo_path / 'weights'
    conf.model = weight_dir / 'v2_SNOB_large_g05_RAFT/wraft_weights-ep01-end.pth'
    conf.add_module_to_statedict = True
    conf.non_strict_loading = False

    conf.iters = 12
    conf.padding_mode = 'nopad'

    conf.name = Path(__file__).stem

    return conf

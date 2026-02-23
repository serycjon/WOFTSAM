from pathlib import Path

from flatsam.config import Config, load_config
from flatsam.dino_features import DINOFeatureExtractor

def get_config():
    # config is the same as _v2, code changed a bit...
    conf = Config()

    # conf.flow = None
    conf.sam = load_config('configs/sam/DAM4SAM.py')


    conf.hough_lines = Config()
    conf.hough_lines.enabled = True
    conf.hough_lines.intersection_corners = False

    conf.resolve_symmetry = Config()
    conf.resolve_symmetry.enabled = True
    conf.resolve_symmetry.only_when_lost = True
    conf.resolve_symmetry.N_correspondences_enough = 4
    conf.resolve_symmetry.resolution_hw = (224, 224)
    conf.resolve_symmetry.fullfit = True
    conf.resolve_symmetry.method = 'dino'
    conf.resolve_symmetry.extractor = DINOFeatureExtractor()
    conf.resolve_symmetry.dino_layer = 11  # 0 - 11, reasonable places 3, 7, 11?
    conf.resolve_symmetry.abs_thr = 0.9
    conf.resolve_symmetry.fts_thr = 0.2
    conf.resolve_symmetry.run_length = 5

    conf.resolve_symmetry.template_update = True

    conf.name = Path(__file__).stem
    return conf

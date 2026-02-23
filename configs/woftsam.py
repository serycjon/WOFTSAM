from pathlib import Path

from flatsam.config import load_config
from flatsam.woft_wrapper import woftsam_track

def get_config():
    conf = load_config('configs/woft/woft.py')
    conf.sam = load_config('configs/sam/SAMbm5.py')
    conf.track_function = woftsam_track
    conf.flatsam = load_config('configs/sam_h.py')

    conf.fallback_robust = True
    conf.fallback_nolost = True

    conf.name = Path(__file__).stem
    return conf

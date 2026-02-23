from pathlib import Path

from flatsam.config import Config, load_config
from flatsam.woft_wrapper import woft_track

def get_config():

    conf = load_config('configs/woft/woft.py')
    conf.sam = load_config('configs/sam/SAMbm5.py')
    conf.track_function = woft_track

    conf.name = Path(__file__).stem
    return conf

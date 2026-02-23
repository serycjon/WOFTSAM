from pathlib import Path

from flatsam.config import load_config
from flatsam.woft_wrapper import woftsam_track
import einops
import torch
from flatsam.utils.least_squares_H import torch_proj_errors

def redet_success_fn(H_prewarped2init, template_coords, cur_pw_coords, weights):
    errs = torch_proj_errors(H_prewarped2init,
                             einops.rearrange(cur_pw_coords, 'xy N -> 1 xy N', xy=2),
                             einops.rearrange(template_coords, 'xy N -> 1 xy N', xy=2))
    inliers = errs <= 5
    inlier_frac = torch.mean(inliers.float())
    return inlier_frac > 0.3

def get_config():
    conf = load_config('configs/woft/woft.py')
    conf.sam = load_config('configs/sam/SAMbm5.py')
    conf.track_function = woftsam_track
    conf.flatsam = load_config('configs/lines_match_intersect_H_whenlost_dino_run5_Tupd_v2.py')

    conf.fallback_robust = True
    conf.fallback_nolost = True

    conf.redet_success_fn = redet_success_fn

    conf.name = Path(__file__).stem
    return conf

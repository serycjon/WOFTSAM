# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import os
import argparse
from pathlib import Path
import logging

import numpy as np
import einops

from flatsam.config import load_config
from flatsam.flatsam import flatsam_track, get_predictor
from flatsam.utils.geom import H_warp
import flatsam.utils.vis as vu
from flatsam.utils.io import GeneralVideoCapture, VideoWriter

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--config', help='path to tracker config file', type=Path,
                        default=Path('configs/woftsam.py'))
    parser.add_argument('--gpu', help='cuda device') 
    args = parser.parse_args()

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


def run(args):
    conf = load_config(args.config)
    sam_predictor = get_predictor(conf.sam.size, conf.sam.memory_stride, conf.sam.do_not_update_when_not_present)

    # load the video / images from a directory
    frames = []

    video_writer = VideoWriter('demo_out.mp4')
    src = 'demo/V24_7.avi'

    # initial coordinates: 4 points, x-coordinate in first row
    # these taken from POT-210 init of V24_7
    init_coords = np.array([[435.0, 732.0, 735.0, 435.0],
                            [318.0, 316.0, 459.0, 453.0]])

    cap = GeneralVideoCapture(src)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    seq_name = 'demo'

    track_function = conf.track_function if conf.track_function else flatsam_track

    all_corners = []

    for frame_i, sam_mask, _, info in track_function(sam_predictor, conf, frames, init_coords, seq_name):
        frame = frames[frame_i]
        try:
            if 'output_H' in info:
                H_init2current = info['output_H']
            else:
                H_init2current = np.linalg.inv(info['output_H2init'])
            current_corners = H_warp(H_init2current, init_coords)
        except Exception:
            # just reuse the last pose if the current is faulty (singular H matrix)
            current_corners = all_corners[-1].copy()

        vis = vu.draw_corners(vu.to_gray_3ch(frame), current_corners, vu.RED)
        video_writer.write(vis)

        all_corners.append(einops.rearrange(current_corners, 'xy N -> 1 (N xy)', xy=2, N=4)) # x, y, x, y, ...

    video_writer.close()
    return 0

def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    results = main()

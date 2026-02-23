# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import os
import argparse
from pathlib import Path
import logging
import gzip
import pickle

import numpy as np
import cv2
import tqdm
import einops
import torch

from flatsam.config import load_config
from flatsam.flatsam import flatsam_track, get_predictor
from flatsam.utils.geom import H_warp
import flatsam.utils.evaluation as eu

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--dataset_base', help='', type=Path)
    parser.add_argument('--dataset', help='', choices=['pot', 'pt', 'poic'], default='pt')
    parser.add_argument('out', help='output directory', type=Path)
    parser.add_argument('--conf', '-c', help='path to flatsam config', type=Path, required=True)
    parser.add_argument('--force', help='force recomputation', action='store_true')
    parser.add_argument('--gpu', help='cuda device') 
    parser.add_argument('--start_from', help='start from Nth sequence', metavar='N', type=int)
    parser.add_argument('--split', help='train, test, or a path to split file')
    parser.add_argument('--debug', help='show debug visualizations', action='store_true')
    parser.add_argument('--seq', help='only do sequence with the given number', type=int)
    parser.add_argument('--nth_seq', help='only do the n-th sequence in the given split', type=int)
    parser.add_argument('--initfix', help='path to reannotated GT initialization', type=Path)
    parser.add_argument('--nonstop', help='', action='store_true')
    parser.add_argument('--debugout', help='write to "debug" config output directory',
                        action='store_true')
    parser.add_argument('--fastforward', '-ff', help='fastforward to frame number', type=int)

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args


def run(args):
    dataset_base = eu.default_base_dirs[args.dataset] if args.dataset_base is None else args.dataset_base
    seq_names = eu.get_sequences[args.dataset](dataset_base, args.split)

    conf = load_config(args.conf)

    if args.dataset != 'pt':
        conf.name = f"{args.dataset}_{conf.name}"

    if args.initfix is not None:
        conf.name = f"initfix_{conf.name}"

    if args.debugout:
        out_dir = args.out / 'debug'
    else:
        out_dir = args.out / conf.name / 'masks'
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = get_predictor(conf.sam.size, conf.sam.memory_stride, conf.sam.do_not_update_when_not_present)

    is_complete = dict(pot=is_complete_pot, pt=is_complete_pt, poic=is_complete_poic)[args.dataset]
    for seq_i, seq_name in enumerate(tqdm.tqdm(seq_names, desc="SEQ")):
        if args.dataset == 'pt':
            seq_num = int(seq_name.split('_')[1], base=10)
            if args.seq is not None and seq_num != args.seq:
                continue
            if args.start_from is not None and seq_num < args.start_from:
                logger.warning(f"Skipping sequence {seq_name}. Just for faster debugging.")
                continue

        if args.nth_seq is not None and seq_i != args.nth_seq:
            continue

        pt_result_file = out_dir / seq_name / f'{seq_name}_FLATSAM.txt'
        if is_complete(dataset_base, seq_name, out_dir, pt_result_file) and not args.force:
            logger.info(f"Skipping {seq_name} - already computed. Use --force to recompute.")
            continue

        logger.debug(f"Starting to track {seq_name}")
        all_corners = []
        init_corners = None

        try:
            for frame_i, mask, frame, debug_info in flatsam_track_dataset_sequence(
                    predictor, conf, args.dataset, dataset_base, seq_name,
                    initfix_dir=args.initfix,
                    debug=args.debug, debug_fastforward=args.fastforward):
                seq_out_dir = out_dir / seq_name
                seq_out_dir.mkdir(parents=True, exist_ok=True)

                # write SAM2.1 mask
                out_path = seq_out_dir / f'{frame_i + 1:05d}.png'
                cv2.imwrite(str(out_path), np.uint8(mask) * 255)

                debug_info['frame_i'] = frame_i
                debug_info['seq'] = seq_name

                # write debug info
                out_path = seq_out_dir / f'{frame_i + 1:05d}.pklz'
                with gzip.open(out_path, 'wb') as fout:
                    pickle.dump(debug_info, fout)

                # store the output poses in the benchmark format
                if frame_i == 0:
                    init_corners = debug_info['init_coords']

                try:
                    if 'output_H' in debug_info:
                        H_init2current = debug_info['output_H']
                    else:
                        H_init2current = np.linalg.inv(debug_info['output_H2init'])
                    current_corners = H_warp(H_init2current, init_corners)
                    all_corners.append(einops.rearrange(current_corners, 'xy N -> 1 (N xy)', xy=2, N=4)) # x, y, x, y, ...
                except Exception:
                    # just reuse the last pose if the current is faulty (singular H matrix)
                    all_corners.append(all_corners[-1])


            # write output pose corners in the official benchmark format
            if len(all_corners) > 0:
                results = np.concatenate(all_corners, axis=0)
                np.savetxt(pt_result_file, results, fmt='%.6f')
        except Exception:
            logger.exception("Tracker failed")
            if not args.nonstop:
                raise

    return 0


def is_complete_pt(planartrack_base, seq_name, base_out_dir, pt_out_file):
    src_dir = planartrack_base / 'sequences' / seq_name
    src_image_paths = sorted([x for x in src_dir.glob('*') if x.is_file()])

    out_dir = base_out_dir / seq_name
    out_image_paths = sorted([x for x in out_dir.glob('*.png') if x.is_file()])

    # return len(src_image_paths) == len(out_image_paths)
    try:
        all_masks_present = all([src_path.stem == out_path.stem
                                 for src_path, out_path in zip(src_image_paths, out_image_paths, strict=True)])
    except ValueError:
        return False

    try:
        pt_out = np.loadtxt(pt_out_file)
        pt_out_complete = pt_out.shape == (len(src_image_paths), 8)
    except Exception:
        return False

    return all_masks_present and pt_out_complete

def is_complete_pot(planartrack_base, seq_name, base_out_dir, pt_out_file):
    src_dir = planartrack_base / 'extracted' / 'images' / seq_name
    src_image_paths = sorted([x for x in src_dir.glob('*') if x.is_file()])

    out_dir = base_out_dir / seq_name
    out_image_paths = sorted([x for x in out_dir.glob('*.png') if x.is_file()])

    # return len(src_image_paths) == len(out_image_paths)
    try:
        all_masks_present = len(src_image_paths) == len(out_image_paths)
        # all_masks_present = all([src_path.stem == out_path.stem
        #                          for src_path, out_path in zip(src_image_paths, out_image_paths, strict=True)])
    except ValueError:
        return False

    try:
        pt_out = np.loadtxt(pt_out_file)
        pt_out_complete = pt_out.shape == (len(src_image_paths), 8)
    except Exception:
        return False

    return all_masks_present and pt_out_complete

def is_complete_poic(dataset_base, seq_name, base_out_dir, pt_out_file):
    src_dir = dataset_base / 'sequences' / seq_name
    src_image_paths = sorted([x for x in src_dir.glob('*') if x.is_file() and x.suffix != '.txt'])

    out_dir = base_out_dir / seq_name
    out_image_paths = sorted([x for x in out_dir.glob('*.png') if x.is_file()])

    # return len(src_image_paths) == len(out_image_paths)
    try:
        all_masks_present = len(src_image_paths) == len(out_image_paths)
        # all_masks_present = all([src_path.stem == out_path.stem
        #                          for src_path, out_path in zip(src_image_paths, out_image_paths, strict=True)])
    except ValueError:
        return False

    try:
        pt_out = np.loadtxt(pt_out_file)
        pt_out_complete = pt_out.shape == (len(src_image_paths), 8)
    except Exception:
        return False

    result = all_masks_present and pt_out_complete
    if not result:
        print(f"{all_masks_present=}")
        if not all_masks_present:
            print(f"{len(src_image_paths)=}")
            print(f"{len(out_image_paths)=}")
        print(f"{pt_out_complete=}")
        if not pt_out_complete:
            print(f"{pt_out.shape=}")
            print(f"({len(src_image_paths)=}, 8)")

    return all_masks_present and pt_out_complete


def flatsam_track_dataset_sequence(sam_predictor, conf, dataset, dataset_base, seq_name, initfix_dir=None, debug=False, debug_fastforward=None):
    if debug:
        print(f'Tracking on "{seq_name}".')

    image_paths = eu.get_img_paths[dataset](dataset_base, seq_name)

    frames = []
    for path in image_paths:
        img = cv2.imread(str(path))
        frames.append(img)

    gt, valid = eu.load_gt_funs[dataset](dataset_base, seq_name)

    init_coords = gt[0, :, :]

    if initfix_dir is not None:
        initfix_path = initfix_dir / f'{seq_name}.txt'

        init_coords = einops.rearrange(np.loadtxt(initfix_path),
                                       '(N xy) -> xy N', xy=2, N=4)

    track_function = flatsam_track
    if conf.track_function:
        track_function = conf.track_function

    yield from track_function(sam_predictor, conf, frames, init_coords, seq_name, debug, debug_fastforward)


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    results = main()

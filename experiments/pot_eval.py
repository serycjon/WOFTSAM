# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import os
import argparse
from pathlib import Path
import logging
import pickle

import einops
import numpy as np
# import cv2
from shapely import Polygon
import tqdm

# import flatsam.utils.vis as vu


logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--pot_base', help='', default=Path("/mnt/datasets/POT-210/"), type=Path)
    parser.add_argument('src', help='directory with masks ("out" from pt_sam_run.py)', type=Path)
    parser.add_argument('out', help='output pickle path', type=Path)
    parser.add_argument('--gpu', help='cuda device') 
    parser.add_argument('--method', help='method name', default='FLATSAM')
    parser.add_argument('--reannot', help='use WOFT reannotation', action='store_true')

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
    seq_names = [path.stem for path in sorted((args.pot_base / 'extracted' / 'images').glob('*'))]

    all_errors = []
    all_BR_errors = []
    all_ious = []
    results = []

    # WOFT_reannotation: all reannotations
    # WOFT_reannotation_coarse: valid_frames = [0, 81, 171, 251, 331, 411]
    # WOFT_reannotation_coarse_onlychanged: only when reannotation changed from orig (apparently used in WOFT paper)
    # see [[id:6dbddc82-8b5f-450e-aa23-75fa64809648][Reproducing old WOFT results]]
    annot_directory = 'WOFT_reannotation_coarse_onlychanged' if args.reannot else 'annotation'
    per_type_p5 = [[] for _ in range(7)]
    per_type_p15 = [[] for _ in range(7)]
    for seq_name in tqdm.tqdm(seq_names, desc="SEQ"):
        if args.method in ['FLATSAM', 'COTRACKER']:
            src_dir = args.src / seq_name
        else:
            src_dir = args.src
        result_path = src_dir / f'{seq_name}_{args.method}.txt'
        if not is_complete(args.pot_base, seq_name, result_path):
            logger.warning(f"Skipping {seq_name} - results not complete.")
            continue

        tracker_outputs = einops.rearrange(np.loadtxt(result_path),
                                           'frames (N xy) -> frames N xy', N=4, xy=2)

        annot = einops.rearrange(
            np.loadtxt(args.pot_base / 'annotation' / annot_directory / f'{seq_name}_gt_points.txt'),
            'frames (N xy) -> frames N xy', N=4, xy=2)

        assert tracker_outputs.shape == annot.shape

        flags = np.loadtxt(args.pot_base / 'annotation' / annot_directory / f'{seq_name}_flag.txt')
        assert flags.shape[0] == annot.shape[0]

        valid_frames = np.any(einops.rearrange(annot, 'frames N xy -> frames (N xy)') != 0, axis=1) & (flags == 0)
        valid_frames[0] = False  # ignore the init frame

        errors = np.sqrt(
            einops.reduce(np.square(annot[valid_frames, :, :] - tracker_outputs[valid_frames, :, :]),
                          'frames N xy -> frames', reduction='sum', xy=2, N=4) / 4)

        best_rot_errors, best_rot = get_best_rotation_errors(annot[valid_frames, :, :], tracker_outputs[valid_frames, :, :])

        ious = compute_ious(annot[valid_frames, :, :], tracker_outputs[valid_frames, :, :])

        seq_results = dict(seq=seq_name, errors=errors, valid_frames=valid_frames, ious=ious,
                           best_rot_errors=best_rot_errors, best_rot=best_rot)
        for thr in [5, 15]:
            seq_results[f'p@{thr}'] = np.nanmean(errors <= thr)
            seq_results[f'BR p@{thr}'] = np.nanmean(best_rot_errors <= thr)

        if args.verbose:
            print(f'| {seq_name} | ', end='')
            for thr in [5, 15]:
                print(f'p@{thr}: {100 * seq_results[f"p@{thr}"]:.1f} | ', end='')
                print(f'BR p@{thr}: {100 * seq_results[f"BR p@{thr}"]:.1f} | ', end='')
            print()

        results.append(seq_results)
        all_errors.append(errors)
        all_BR_errors.append(best_rot_errors)
        all_ious.append(ious)

        seq_type = int(seq_name.split('_')[1]) - 1
        per_type_p5[seq_type].append(seq_results['p@5'])
        per_type_p15[seq_type].append(seq_results['p@15'])
            
    errors = np.concatenate(all_errors, axis=0)
    BR_errors = np.concatenate(all_BR_errors, axis=0)
    for thr in [5, 15]:
        print(f'p@{thr}: {100 * np.mean(errors <= thr):.1f} | ', end='')
        print(f'BR p@{thr}: {100 * np.mean(BR_errors <= thr):.1f} | ', end='')

    ious = np.concatenate(all_ious, axis=0)
    print(f'corner IOU: {100 * np.mean(ious):.1f} | ', end='')
    print(f'corner IOU > 80: {100 * np.mean(ious > 0.8):.1f} | ', end='')
    print(f'corner IOU < 50: {100 * np.mean(ious < 0.5):.1f} | ', end='')
    print()

    type_names = ['scale', 'rotation', 'perspective', 'blur', 'occlusion', 'out-of-view', 'unconstrained']
    for seq_type, p5, p15 in zip(type_names, per_type_p5, per_type_p15):
        print(f'| {seq_type} | {100 * np.nanmean(p5):0.1f} | {100 * np.nanmean(p15):0.1f} |')
        

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('wb') as fout:
        pickle.dump(results, fout)
    print(f'Results written to {args.out}')
    return 0

def is_complete(pot_base, seq_name, tracker_result_path):
    try:
        tracker_outputs = np.loadtxt(tracker_result_path)
    except Exception:
        return False

    src_dir = pot_base / 'extracted' / 'images' / seq_name
    src_image_paths = sorted([x for x in src_dir.glob('*') if x.is_file()])

    return tracker_outputs.shape[0] == len(src_image_paths)

def compute_ious(gt, pred):
    """Compute IOUs on polygons

    args:
        gt: (N, 4, xy) array
        pred: (N, 4, xy) array

    returns:
        ious: (N, ) array
    """
    ious = []

    for i in range(gt.shape[0]):
        gt_poly = Polygon(gt[i, :, :])
        pred_poly = Polygon(pred[i, :, :])
        try:
            assert pred_poly.is_valid
            intersection_poly = gt_poly.intersection(pred_poly)
            intersection = intersection_poly.area
            union = gt_poly.area + pred_poly.area - intersection

            if union == 0:
                iou = 1
            else:
                iou = intersection / union
        except Exception:
            iou = 0

        ious.append(iou)
    return np.array(ious)

def main():
    args = parse_arguments()
    return run(args)

def get_best_rotation_errors(gts, preds):
    all_errors = []
    for shift in range(4):
        errors = np.sqrt(
            einops.reduce(np.square(gts - np.roll(preds, shift, axis=1)),
                          'frames N xy -> frames', reduction='sum', xy=2, N=4) / 4)
        all_errors.append(errors)
    all_errors = np.array(all_errors)
    best_errors = einops.reduce(all_errors, 'shifts frames -> frames', reduction='min')
    best_shift = np.argmin(all_errors, axis=0)
    return best_errors, best_shift


if __name__ == '__main__':
    results = main()

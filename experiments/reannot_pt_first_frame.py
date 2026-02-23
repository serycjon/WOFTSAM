# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import sys
import argparse
from pathlib import Path
import logging

import einops
import numpy as np
import cv2

import flatsam.utils.geom as gu
import flatsam.utils.vis as vu
import flatsam.utils.evaluation as eu

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--dataset_base', help='', type=Path)
    parser.add_argument('--dst', help='', type=Path, default=Path('./PlanarTrack_init_reannotation/'))
    parser.add_argument('--split', help='train, test, or a path to split file', default='test')
    
    args = parser.parse_args()

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args


def run(args):
    out_dir = args.dst
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_base = eu.default_base_dirs['pt'] if args.dataset_base is None else args.dataset_base
    seq_names = eu.get_sequences['pt'](dataset_base, args.split)

    # seq_name = seq_names[13]
    # annot, valid_frames = eu.pt_pos_gt(dataset_base, seq_name)
    gui = ReannotGUI(seq_names,
                     lambda seq_name: eu.pt_pos_gt(dataset_base, seq_name),
                     lambda seq_name: eu.pt_img_paths(dataset_base, seq_name),
                     out_dir)
    gui.run()

    return 0


def main():
    args = parse_arguments()
    return run(args)

def select_large_reference(annot, valid_frames, restrict_to_hw=None):
    N = annot.shape[0]

    best_area = 0
    best_i = None
    for i in range(1, N):
        if valid_frames[i]:
            corners = annot[i, :, :]
            area, fully_inside = gu.polygon_area(corners, restrict_to_hw=restrict_to_hw)

            if fully_inside and area > best_area:
                best_area = area
                best_i = i
    return best_i

def load_init(directory, seq_name, default):
    path = directory / f'{seq_name}.txt'
    try:
        assert path.exists()
        init = np.loadtxt(path)
        init = einops.rearrange(init, '(N xy) -> xy N', xy=2, N=4)
    except Exception:
        logger.exception("Failed to load saved init")
        init = default
    return init

def save_init(directory, seq_name, init):
    path = directory / f'{seq_name}.txt'
    np.savetxt(path, einops.rearrange(init, 'xy N -> 1 (N xy)', xy=2, N=4), fmt='%.4f')
    print(f'init saved to {path}')

class ReannotGUI():
    def __init__(self, seq_names, gt_load_fn, img_paths_fn, out_dir):
        self.seq_names = seq_names
        self.load_gt = gt_load_fn
        self.get_img_paths = img_paths_fn
        self.out_dir = out_dir
        self.seq_i = 0

    def run(self):
        self.load_seq(self.seq_i)

        while True:
            self.display()
            while True:
                c = cv2.waitKey(0)
                if c == ord('q'):
                    self.save()
                    sys.exit(1)
                elif c == ord(' '):
                    break
                elif c == 85: # pageup
                    self.set_reference_frame(self.reference_frame_i - 1)
                    break
                elif c == 86: # pagedown
                    self.set_reference_frame(self.reference_frame_i + 1)
                    break
                elif c == 80: # home
                    self.save()
                    self.load_seq(self.seq_i - 1)
                    break
                elif c == 87: # end
                    self.save()
                    self.load_seq(self.seq_i + 1)
                    break
                elif c == ord('0'):
                    self.set_control_point(0)
                    break
                elif c == ord('1'):
                    self.set_control_point(1)
                    break
                elif c == ord('2'):
                    self.set_control_point(2)
                    break
                elif c == ord('3'):
                    self.set_control_point(3)
                    break
                elif c == 82: # up
                    self.move_control_point(0, -1)
                    break
                elif c == 81: # left
                    self.move_control_point(-1, 0)
                    break
                elif c == 84: # down
                    self.move_control_point(0, 1)
                    break
                elif c == 83: # right
                    self.move_control_point(1, 0)
                    break
                elif c == ord('-'):
                    self.move_step *= 2
                elif c == ord('+'):
                    self.move_step /= 2
                elif c == ord('/'):
                    self.align_mode = 'init'
                    break
                elif c == ord('*'):
                    self.align_mode = 'reference'
                    break
                elif c == 13: # return
                    self.align_mode = 'align'
                    break
                else:
                    print(c)
                    print(chr(c))

    def load_seq(self, seq_i):
        N_seq = len(self.seq_names)
        if seq_i < 0:
            seq_i = 0
        if seq_i >= N_seq:
            seq_i = N_seq - 1
        seq_name = self.seq_names[seq_i]
        annot, valid_frames = self.load_gt(seq_name)
        img_paths = self.get_img_paths(seq_name)
        template_img = cv2.imread(str(img_paths[0]))
        reference_frame_i = select_large_reference(annot, valid_frames, template_img.shape[:2])

        self.seq_i = seq_i
        self.img_paths = img_paths
        self.annot, self.valid_frames = annot, valid_frames
        self.set_reference_frame(reference_frame_i)
        self.template_img = template_img
        self.current_control_point = 0
        self.move_step = 1
        self.align_mode = 'align'
        self.init_corners = load_init(self.out_dir, seq_name, self.annot[0, :, :].copy())

    def save(self):
        save_init(self.out_dir, self.seq_names[self.seq_i], self.init_corners)

    def set_reference_frame(self, i):
        N_frames = self.annot.shape[0]
        if i < 0:
            i = 0
        if i >= N_frames:
            i = N_frames - 1
        self.reference_frame_i = i
        self.reference_frame = cv2.imread(str(self.img_paths[i]))
        self.reference_corners = self.annot[i, :, :].copy()
        self.reference_vis = vu.draw_corners(self.reference_frame, self.reference_corners,
                                             vu.GREEN, thickness=1, with_cross=False, with_TL=False,
                                             lineType=cv2.LINE_AA)
        self.reference_vis = vu.draw_text(self.reference_vis,
                                          f"{self.seq_names[self.seq_i]}: #{self.reference_frame_i}",
                                          pos='tl', thickness=1, size=1)

    def set_control_point(self, i):
        assert i >= 0
        assert i < 4
        self.current_control_point = i

    def move_control_point(self, dx, dy):
        self.init_corners[:, self.current_control_point] += (dx * self.move_step,
                                                             dy * self.move_step)

    def display(self):
        init = vu.draw_corners(self.template_img, self.init_corners, vu.RED, thickness=1, with_cross=False, with_TL=False, lineType=cv2.LINE_AA)
        init = vu.circle(init, self.init_corners[:, self.current_control_point], 2, vu.WHITE, lineType=cv2.LINE_AA)
        vu.imshow("cv: init", init)
        vu.imshow("cv: reference", self.reference_vis)
        H, _ = cv2.findHomography(einops.rearrange(self.init_corners, 'xy N -> N 1 xy', xy=2, N=4),
                                  einops.rearrange(self.reference_corners, 'xy N -> N 1 xy', xy=2, N=4))
        try:
            warped_init = cv2.warpPerspective(self.template_img, H, (self.template_img.shape[1], self.template_img.shape[0]))
            if self.align_mode == 'align':
                alignment = vu.vis_alignment_plain(warped_init, self.reference_frame, equalize_hist=True)
            elif self.align_mode == 'init':
                alignment = warped_init
            elif self.align_mode == 'reference':
                alignment = self.reference_frame
        except cv2.error:
            alignment = np.zeros_like(self.template_img)
        vu.imshow("cv: alignment", alignment)

if __name__ == '__main__':
    results = main()

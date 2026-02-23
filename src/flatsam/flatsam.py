# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import sys
from pathlib import Path
import logging
from timeit import default_timer as timer
import math
import itertools
import pickle

import cv2
import numpy as np
import einops
import torch
from sam2.build_sam import build_sam2_video_predictor
from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt
import scipy
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity

from flatsam.utils.io import ram_directory_of_images
import flatsam.utils.vis as vu
from flatsam.utils.misc import remap, col_enumerate
import numpy as np
import flatsam.utils.geom as gu
from flatsam.utils.geom import H_compose, H_warp, H_warp_lines
from flatsam.utils.geom import find_TRS, find_T
from flatsam.utils.timing import general_time_measurer


logger = logging.getLogger(__name__)
ALL_COLORS = [
    vu.hex_to_bgr(c)
    for c in ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']]

class NextSequence(Exception):
    pass

def get_predictor(size, memory_stride=None, do_not_update_when_not_present=False):
    logger.info("Building SAM2 video predictor")
    options = {'tiny': ('tiny', 't'),
               'small': ('small', 's'),
               'base_plus': ('base_plus', 'b+'),
               'large': ('large', 'l')
               }
    checkpoint = f"./checkpoints/sam2.1_hiera_{options[size][0]}.pt"
    model_cfg = f"configs/sam2.1/sam2.1_hiera_{options[size][1]}.yaml"
    hydra_overrides_extra = []
    if memory_stride is not None:
        hydra_overrides_extra.append(f"++model.memory_temporal_stride_for_eval={memory_stride}")
    if do_not_update_when_not_present:
        hydra_overrides_extra.append("++model.do_not_update_when_not_present=True")

    return build_sam2_video_predictor(model_cfg, checkpoint,
                                      hydra_overrides_extra=hydra_overrides_extra)

def inflate(coords, amount):
    center = einops.reduce(coords, 'xy N -> xy', reduction='mean', xy=2)
    H_to_origin = np.array([[1, 0, -center[0]],
                            [0, 1, -center[1]],
                            [0, 0, 1]])
    H_scale = np.array([[1 + amount, 0, 0],
                        [0, 1 + amount, 0],
                        [0, 0, 1]])
    H = H_compose(H_to_origin, H_scale, np.linalg.inv(H_to_origin))
    return H_warp(H, coords)

def flatsam_track(predictor, conf, frames, init_coords, seq_name, debug=False, debug_fastforward=None):
    sequence_timer = general_time_measurer('flatsam_sequence_tracking', cuda_sync=False, start_now=True)
    init_mask = vu.draw_mask(init_coords, frames[0].shape[:2])
    inflated_init_mask = vu.draw_mask(inflate(init_coords, 0.1), frames[0].shape[:2])

    sam_on_first_frame = False
    sam_timer = general_time_measurer('sam_tracking', cuda_sync=False, start_now=False)
    init_frame = frames[0].copy()
    H2init = np.eye(3)
    init_HAs = None  # initial full Affine from canonical to the init image
    init_angles = None  # initial inner rotations (each corresponding
                        # to one corner of the init mask being on
                        # right (angle = 0) in the cannonical frame)

    H2init_final = np.eye(3)
    last_mask_corners = None
    best_symmetry_shift_run = [0, 0]
    init_center_of_mass = None
    previously_not_enough_correspondences = False
    best_template_frame = 0
    best_template_H_to_init = np.eye(3)
    best_template_score = 0
    lost_at_least_once = False

    sam_timer.start()
    for frame_idx, mask in sam_track(frames, init_mask, predictor, seq_name=seq_name,
                                     estimate_on_init_frame=sam_on_first_frame):
        sam_timer.stop()

        debug_info = {'frame_idx': frame_idx, 'debug_enabled': debug}
        if frame_idx == 0:
            debug_info['size'] = frames[0].shape
            debug_info['init_coords'] = init_coords.copy()

        frame = frames[frame_idx]

        line_based = conf.hough_lines.enabled
        if line_based:
            hough_timer = general_time_measurer('contour_hough', cuda_sync=False, start_now=True)
            contours_orig, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours_orig = [contour for contour in contours_orig if contour.shape[0] > 20]
            contours = [einops.rearrange(contour, 'N 1 xy -> xy N', xy=2) for contour in contours_orig]
            on_boundaries = [np.logical_or.reduce(
                (approx_touching_edges(contour + np.array([[0], [1]]), mask.shape, margin=1),
                 approx_touching_edges(contour + np.array([[0], [-1]]), mask.shape, margin=1),
                 approx_touching_edges(contour + np.array([[1], [0]]), mask.shape, margin=1),
                 approx_touching_edges(contour + np.array([[-1], [0]]), mask.shape, margin=1),
                 ))
                             for contour in contours]

            hough_lines = contour_hough(mask, contours, on_boundaries, conf.hough_lines)
            hough_timer.stop()
            # hough_lines_refined = hough_lines
            refine_timer = general_time_measurer('lines_refinement', cuda_sync=False, start_now=True)
            hough_lines_refined, img_lines_pts = refine_lines(hough_lines, contours, on_boundaries, conf.hough_lines)
            refine_timer.stop()
            postprocess_timer = general_time_measurer('lines_postprocessing', cuda_sync=False, start_now=True)

            # filter out lines with zero support
            line_mask = [pts.size > 0 for pts in img_lines_pts]
            hough_lines_refined = hough_lines_refined[:, line_mask]
            img_lines_pts = [pts for i, pts in enumerate(img_lines_pts) if line_mask[i]]

            endpoints_img = [line_to_segment(img_line, img_pts) for img_line, img_pts in zip(hough_lines_refined.transpose(), img_lines_pts)]

            # filter out short lines
            short_lines = np.array([line_segment_length(segment) < 5 for segment in endpoints_img], dtype=bool)
            hough_lines_refined = hough_lines_refined[:, ~short_lines]
            img_lines_pts = [pts for pts, short in zip(img_lines_pts, short_lines) if not short]

            debug_info['hough_lines_refined'] = hough_lines_refined

            intersections, intersection_parents = lines_to_points(hough_lines_refined)
            intersections = gu.p2e(intersections)
            valid_intersections, valid_intersections_mask = keep_only_points_close_to_contours(intersections, contours)
            point_matches = bruteforce_pointset_matching(init_coords, valid_intersections)
            valid_parents = [x for x, valid in zip(intersection_parents, valid_intersections_mask) if valid]
            valid_point_matches = discard_diagonal_matches(point_matches, valid_parents, N_template_points=init_coords.shape[1])

            best_H, best_diff, best_ids = find_closest_H(init_coords, valid_intersections, np.linalg.inv(H2init), valid_point_matches)
            postprocess_timer.stop()
            N_correspondences = len(best_ids[0])
            # print(f"{best_ids=}")
            # print(f"{best_diff=}")

            debug_info['best_line_matching_H'] = best_H

            if conf.resolve_symmetry.enabled:
                symmetry_timer = general_time_measurer('symmetry_resolution', cuda_sync=True, start_now=True)
                if conf.resolve_symmetry.only_when_lost:
                    if N_correspondences < conf.resolve_symmetry.N_correspondences_enough:
                        previously_not_enough_correspondences = True
                        best_symmetry_shift_run = [0, 0]
                        lost_at_least_once = True

                    if previously_not_enough_correspondences and N_correspondences == conf.resolve_symmetry.N_correspondences_enough:
                        best_H, init_coords, init_frame, undo_info = swap_better_template(best_H, init_coords, frames,
                                                                                          best_template_frame, best_template_H_to_init)
                        best_H, symmetry_resolve_shift, shift_confirmed, debug_info = resolve_symmetry_img(
                            best_H, init_coords, best_symmetry_shift_run,
                            init_frame, frame, conf, debug_info)
                        best_H, init_coords, init_frame = undo_swap_better_template(undo_info, best_H)
                        if shift_confirmed:
                            previously_not_enough_correspondences = False
                else:
                        best_H, symmetry_resolve_shift, shift_confirmed, debug_info = resolve_symmetry_img(
                            best_H, init_coords, best_symmetry_shift_run,
                            init_frame, frame, conf, debug_info)
                symmetry_timer.stop()

                            

            H_prewarp = np.linalg.inv(best_H)

        debug_info['H_prewarp'] = H_prewarp.copy()

        lost = None
        H2init = H_prewarp.copy()

        debug_info['H2init'] = H2init.copy()
        H2init_final = H2init.copy() if not lost else H_prewarp.copy()
        debug_info['output_H2init'] = H2init_final.copy()

        if not lost_at_least_once and conf.resolve_symmetry.template_update:
            H_init2current_final = np.linalg.inv(H2init_final)
            current_coords = H_warp(H_init2current_final, init_coords)
            frame_score, is_fully_inside = gu.polygon_area(current_coords, restrict_to_hw=frames[0].shape[:2])
            if frame_score > best_template_score and is_fully_inside:
                best_template_score = frame_score
                best_template_frame = frame_idx
                best_template_H_to_init = H2init_final

                if debug:
                    vu.imshow("cv: template frame", frame, keep_history=True)

        debug_info['lost'] = lost

        tl_vis = frame.copy()
        vis = tl_vis
        if debug:
            # visualization {{{
            color = (0, 0, 255) if not lost else (170, 170, 170)

            # frame = prewarped
            corners = H_warp(np.linalg.inv(debug_info['output_H2init']), init_coords)
            tl_vis = vu.blend_mask(tl_vis, mask > 0, alpha=0.2)
            # tl_vis = vu.draw_corners(tl_vis, H_warp(np.linalg.inv(H_prewarp), init_coords), vu.YELLOW, alpha=0.5)
            # tl_vis = vu.draw_corners(tl_vis, H_warp(np.linalg.inv(HA_current2init), init_coords), vu.GREEN, alpha=0.7)
            show_Aff_ellipse = False
            if show_Aff_ellipse:
                unit = 2
                angles = np.linspace(0, 2 * np.pi, 100)
                circle = unit * np.stack((np.cos(angles), np.sin(angles)), axis=0)
                ellipse = H_warp(HA_canonical2current, circle)
                tl_vis = vu.polylines(tl_vis, [einops.rearrange(ellipse, 'xy N -> N 1 xy', xy=2)], True, vu.MAGENTA, thickness=2)
                tl_vis = vu.circle(tl_vis, ellipse[:, 0], 3, vu.GREEN, thickness=-1)

            all_colors = [
                vu.hex_to_bgr(c)
                for c in ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']]

            # for line, (init_line_i, current_line_i) in zip(hough_lines_refined.transpose(), best_pairing):
            #     if current_line_i is None:
            #         l_color = vu.BLACK
            #     else:
            #         l_color = all_colors[init_line_i]
            #     vu.line_eq(tl_vis, line, color=l_color, thickness=2, lineType=cv2.LINE_AA)

            for line_i, line in col_enumerate(hough_lines_refined):
                vu.line_eq(tl_vis, line, color=all_colors[line_i], thickness=5, lineType=cv2.LINE_AA)

                # for pt in endpoints_img[line_i].transpose():
                #     vu.circle(tl_vis, pt, 10, color=all_colors[line_i], thickness=-1)

            for pt_i, pt in col_enumerate(valid_intersections):
                vu.circle(tl_vis, pt, 20, color=all_colors[pt_i], thickness=-1)

            # for line_i, line_pts in enumerate(img_lines_pts):
            #     if line_pts.size == 0:
            #         continue
            #     tl_vis[line_pts[1, :], line_pts[0, :], :] = all_colors[line_i % len(all_colors)]
                # tl_vis[line_pts[1, :20], line_pts[0, :20], :] = 0
                # tl_vis[line_pts[1, line_pts.shape[1]-20:], line_pts[0, line_pts.shape[1]-20:], :] = 255

            try:
                tr_vis = vu.vis_alignment_plain(prewarped_current, prewarped_template)
            except Exception:
                tr_vis = None
                tr_vis = np.ones_like(tl_vis) * 255
                cv2.drawContours(tr_vis, contours_orig, -1, vu.BLACK, 1)

                for line_i, line_pts in enumerate(img_lines_pts):
                    if line_pts.size == 0:
                        continue
                    tr_vis[line_pts[1, :], line_pts[0, :], :] = all_colors[line_i % len(all_colors)]

            bl_template_vis = init_frame.copy()
            radius = 30
            bl_template_vis = vu.draw_corners(bl_template_vis, init_coords,
                                              vu.RED, with_cross=False, with_TL=True,
                                              thickness=1, TL_rel_scale=4, lineType=cv2.LINE_AA,
                                              draw_ori_color=vu.blend_colors(vu.RED, vu.WHITE, 0.2))

            br_frame_vis = frame.copy()
            br_frame_vis = vu.draw_corners(br_frame_vis, corners, color, with_cross=False, with_TL=True,
                                           thickness=1, TL_rel_scale=4, TL_radius=20, lineType=cv2.LINE_AA,
                                           draw_ori_color=vu.blend_colors(color, vu.WHITE, 0.2))
            for i, pt in col_enumerate(intersections):
                br_frame_vis = vu.circle(br_frame_vis, pt, 7, vu.BLUE, thickness=-1)

            vis = vu.tile([[tl_vis, tr_vis],
                           [bl_template_vis, br_frame_vis]])
            vu.imshow("cv: vis", vu.draw_text(vis, f'#{frame_idx}', pos='tr'), keep_history=True)
            while True:
                c = cv2.waitKey(0 if (debug_fastforward is None or (debug_fastforward > 0 and frame_idx >= debug_fastforward)) else 5)
                if c == ord('q'):
                    sys.exit(1)
                elif c == ord(' '):
                    break
                elif c == ord('n'):
                    return
                elif c == ord('e'):
                    raise RuntimeError("blab")
                elif c == ord('b'):
                    breakpoint()
                elif c == ord('h'):
                    vu.imshow_history_show_older()
                elif c == ord('l'):
                    vu.imshow_history_show_newer()
                elif c == -1:
                    break

            # }}}

        sam_timer.report(reduction='mean')
        if line_based:
            hough_timer.report(reduction='sum')
            refine_timer.report(reduction='sum')
            postprocess_timer.report(reduction='sum')
            if conf.resolve_symmetry.enabled:
                symmetry_timer.report(reduction='sum')
        yield frame_idx, mask, vis, debug_info
        sam_timer.start()

    sequence_timer.stop()
    sequence_timer.report(reduction='mean')


def sam_track(frames, init_mask, predictor, seq_name=None, estimate_on_init_frame=False, bbox_init=False):
    """
    args:
        estimate_on_init_frame: when True, repeat the initial frame and extract the SAM prediction there (instead of reusing exactly the init_mask)
    """
    OBJ_ID = 1
    start_time = timer()

    with ram_directory_of_images(frames, seq_name, double_first_frame=estimate_on_init_frame) as video_path:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(str(video_path), offload_video_to_cpu=True, async_loading_frames=True)

            # add new prompts and instantly get the output on the same frame
            if bbox_init:
                # x_min, y_min, x_max, y_max
                init_bbox = gu.mask2bbox(init_mask).as_xyxy()
                out_frame_idx, out_object_ids, out_mask_logits = predictor.add_new_points_or_box(state, frame_idx=0, obj_id=OBJ_ID, box=init_bbox)
            else:
                out_frame_idx, out_object_ids, out_mask_logits = predictor.add_new_mask(state, frame_idx=0, obj_id=OBJ_ID, mask=init_mask)

            duration_s = timer() - start_time
            logger.debug(f'SAM preparation: {duration_s}s')

            start_time = timer()
            n_timed_frames = 0
            # propagate the prompts to get masklets throughout the video
            last_frame_idx = -1
            sam_timer = general_time_measurer('sam_inner_tracking', cuda_sync=False, start_now=False)
            sam_timer.start()
            for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
                assert frame_idx == last_frame_idx + 1
                last_frame_idx = frame_idx
                n_timed_frames += 1
                if estimate_on_init_frame and frame_idx == 0:
                    continue
                out_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8) > 0
                for oid, mask_logit in zip(object_ids, mask_logits):
                    if oid == OBJ_ID:
                        mask = (einops.rearrange(mask_logit, '1 H W -> H W') > 0).cpu().numpy()
                        out_mask = np.logical_or(out_mask, mask)
                sam_timer.report(reduction='mean')

                with torch.amp.autocast('cuda', enabled=False):
                    yield (frame_idx - 1 if estimate_on_init_frame else frame_idx,
                           out_mask)
                sam_timer.start()

            duration_s = float(timer() - start_time)
            ms_per_frame = (1000 * duration_s) / n_timed_frames
            fps = n_timed_frames / duration_s

            logger.debug(f'SAM tracking on {mask.shape[1]}x{mask.shape[0]} images: {ms_per_frame:.0f} ms/frame, {fps:0.1f} FPS')

def approx_touching_edges(pts, shape, margin=0):
    return np.any((pts <= margin) | (pts >= [[shape[1] - 1 - margin], [shape[0] - 1 - margin]]),
                  axis=0)


def rotate_HA(HpA, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    HA_full = np.matmul(HpA, R)
    return HA_full


def find_transformation_2(init_pts, current_pts, last_H_init2current):
    N_corr = init_pts.shape[1]
    if N_corr >= 4:
        # full homography wrt initial frame
        # logger.debug('estimating full H from mask corners')
        H, _ = cv2.findHomography(einops.rearrange(init_pts, 'xy N -> N 1 xy', xy=2),
                                  einops.rearrange(current_pts, 'xy N -> N 1 xy', xy=2),
                                  method=0)

    elif N_corr > 1:
        # translation, rotation, scale wrt previous frame
        logger.debug(f'estimating TRS H from mask corners ({N_corr=})')
        last_pts = H_warp(last_H_init2current, init_pts)
        TRS = find_TRS(last_pts, current_pts)
        H = H_compose(last_H_init2current, TRS)

    elif N_corr == 1:
        # translation wrt previous frame
        logger.debug(f'estimating Translation H from mask corners ({N_corr=})')
        last_pts = H_warp(last_H_init2current, init_pts)
        T = find_T(last_pts, current_pts)
        H = H_compose(last_H_init2current, T)

    else:
        logger.debug(f'falling back to no motion ({N_corr=})')
        H = last_H_init2current
        # logger.debug(f'falling back to last H + mask center of gravity shift ({N_corr=})')
        # H_shift = np.array([[1, 0, mask_shift[0]],
        #                     [0, 1, mask_shift[1]],
        #                     [0, 0, 1]])
        # H = H_compose(H_shift, last_H2init)

    return H

def get_lengths(pts):
    return np.sqrt(einops.reduce(np.square(pts), 'xy N -> N', xy=2, reduction='sum'))

def contour_hough(mask, img_contours, on_boundaries, conf):
    """Find Hough transform lines in mask contours

    args:
        mask: bool (H, W) numpy array

    returns:
        lines: (abc, N) numpy array of lines, such that ax + by + c = 0 for points on the lines
    """
    show = False

    if len(img_contours) == 0:
        return np.zeros((3, 0))

    all_cnt = np.concatenate(img_contours, axis=1)
    contour_center = np.mean(all_cnt, axis=1, keepdims=True)
    contours = [cnt - contour_center for cnt in img_contours]

    point_distances = [get_lengths(cnt) for cnt in contours]
    max_dist = max(np.amax(dists) for dists in point_distances)
    min_dist = 0

    N_angles = 359
    N_dists = 100

    # def dist_map(x):
    #     return remap(x, min_dist, max_dist, 0, N_dists - 1)

    def dist_unmap(x):
        return remap(x, 0, N_dists - 1, min_dist, max_dist)

    # def angle_map(x):
    #     return remap(x, 0, 2 * np.pi, 0, N_angles - 1)

    def angle_unmap(x):
        return remap(x, 0, N_angles - 1, 0, 2 * np.pi)

    accumulator = np.zeros((N_dists, N_angles))

    N_pts_to_use = 1000000
    contour_step = 4
    degree_range = 10 # +-
    rad_range = np.radians(degree_range)
    n_degree_steps = 11

    delta_angles = np.linspace(-rad_range, +rad_range, n_degree_steps).tolist()

    sampled_pts = []

    for pts, on_boundary in zip(contours, on_boundaries, strict=True):
        N_pts = pts.shape[1]
        contour_step_outer = max(N_pts // N_pts_to_use, 1)
        # contour_step_outer = 1
        for i in range(0, N_pts, contour_step_outer):
            if on_boundary[i]:
                continue

            z = pts[:, ((i - contour_step) + N_pts) % N_pts]
            a = pts[:, i]
            b = pts[:, (i + contour_step) % N_pts]

            if show:
                sampled_pts.append(a)

            diff = (b - z)
            if diff[0] == 0 and diff[1] == 0:
                continue
            angle = np.arctan2(diff[1], diff[0])
            dist = math.cos(angle) * a[1] - math.sin(angle) * a[0]

            norm_angle = angle + np.pi / 2
            norm_to_direction = -np.pi / 2
            # norm_dist = np.abs(dist)
            if dist < 0:
                norm_angle -= np.pi
                norm_to_direction = +np.pi / 2

            for delta_angle in delta_angles:
                res_angle = norm_angle + delta_angle
                res_angle = (res_angle + 2 * np.pi) % (2 * np.pi)
                dir_angle = (res_angle + norm_to_direction).item()
                dist = abs(math.cos(dir_angle) * a[1] - math.sin(dir_angle) * a[0])
                amount = remap(abs(delta_angle), 0, rad_range, 1, 0.5)

                # accumulate
                dist_mapped = remap(dist, min_dist, max_dist, 0, N_dists - 1)
                angle_mapped = remap(res_angle, 0, 2 * np.pi, 0, N_angles - 1)
                row = int(round(dist_mapped.item()))
                row = max(0, min(row, accumulator.shape[0] - 1))
                # row = int(np.clip(np.round(dist_mapped), 0, accumulator.shape[0] - 1))

                col = int(round(angle_mapped.item()))
                col = max(0, min(col, accumulator.shape[1] - 1))
                # col = int(np.clip(np.round(angle_mapped), 0, accumulator.shape[1] - 1))
                accumulator[row, col] += amount

    ## find peaks in periodic signal:
    # repeat 3 times, find peaks, get rid of the two side ones
    cyclic = np.tile(accumulator, (1, 3))
    cyclic = scipy.ndimage.filters.gaussian_filter(cyclic, sigma=4)

    peaks = peak_local_max(cyclic, min_distance=10, threshold_rel=None, exclude_border=False)
    peaks = peaks[(peaks[:, 1] >= accumulator.shape[1]) & (peaks[:, 1] < 2 * accumulator.shape[1]), :]
    peaks[:, 1] -= accumulator.shape[1]

    blurcumulator = cyclic[:, accumulator.shape[1]:2*accumulator.shape[1]]

    ## topk peaks
    peak_values = blurcumulator[peaks[:, 0], peaks[:, 1]]
    topk = 4
    real_topk = min(topk, len(peak_values))
    top_ids = np.argpartition(peak_values, -real_topk)[-real_topk:]
    peaks = peaks[top_ids, :]

    ## sort by the angle
    peaks = np.take_along_axis(peaks, np.argsort(peaks[:, 1, np.newaxis], axis=0), axis=0)
    # print(f'Hough {peaks=}')

    if show:
        # log_blurcumulator = blurcumulator.copy()
        # log_blurcumulator[blurcumulator == 0] = np.nan
        # log_blurcumulator = np.log(log_blurcumulator)
        # vis = vu.cv2_colormap(log_blurcumulator)
        vis = vu.cv2_colormap(accumulator)

        for peak in peaks:
            vis[peak[0], peak[1], :] = vu.RED

        vu.imshow("cv: accumulator", vis, keep_history=True)

        H_undo_centering = np.eye(3)
        H_undo_centering[0, 2] = contour_center[0]
        H_undo_centering[1, 2] = contour_center[1]
        sampled_pts_vis = 255 * np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        sampled_pts = einops.rearrange(np.array(sampled_pts), 'N xy -> xy N', xy=2)
        sampled_pts = H_warp(H_undo_centering, sampled_pts)
        for i, pt in col_enumerate(sampled_pts):
            vu.circle(sampled_pts_vis, pt, 5, vu.RED, thickness=-1)

        vu.imshow("cv: sampled_pts_vis", sampled_pts_vis)
        while True:
            c = cv2.waitKey(0)
            if c == ord('q'):
                sys.exit(1)
            elif c == ord(' '):
                break
        # while True:
        #     c = cv2.waitKey(0)
        #     if c == ord('q'):
        #         sys.exit(1)
        #     elif c == ord(' '):
        #         break

    lines = []
    for peak in peaks:
        distance = dist_unmap(peak[0])
        angle = angle_unmap(peak[1])

        line = np.array([np.cos(angle), np.sin(angle), -distance])
        lines.append(line)

    if len(lines) == 0:
        lines = np.array([[], [], []])
    else:
        lines = einops.rearrange(np.array(lines), 'N abc -> abc N', abc=3)

    lines, _ = recenter_lines(lines, -contour_center.flatten())
    return lines

def refine_lines(lines, contours, on_boundaries, conf):
    """
    args:
        lines: (abc, N) array of lines in homogeneous coordinates
                  (point x is on a line lines[:, 0] if np.dot(lines[:, 0], e2p(x)) == 0)
    """
    N_lines = lines.shape[1]
    # contours, _ = cv2.findContours(mask_img.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours = [einops.rearrange(contour, 'N 1 xy -> xy N', xy=2) for contour in contours]
    # contours = hough_contours
    if N_lines == 0:
        return lines, []

    # on_boundaries = [approx_touching_edges(contour, mask_img.shape, margin=0) for contour in contours]

    normalsT = lines[:2, :].transpose()
    distances = lines[2, :]

    line_pts = [[] for _ in range(N_lines)]

    previous_line_i = None
    line_order = []

    N_pts_to_use = 100000
    contour_step = 4

    for pts, on_boundary in zip(contours, on_boundaries, strict=True):
        N_pts = pts.shape[1]
        contour_step_outer = max(N_pts // N_pts_to_use, 1)
        for i in range(0, N_pts, contour_step_outer):
            if on_boundary[i]:
                continue
            z = pts[:, ((i - contour_step) + N_pts) % N_pts]
            a = pts[:, i]
            b = pts[:, (i + contour_step) % N_pts]

            line_vector = (b - z).astype(np.float64)
            norm = np.hypot(line_vector[0], line_vector[1])
            if norm == 0:
                continue
            line_vector /= norm

            dist_diffs = np.matmul(normalsT, a[:, np.newaxis]).flatten() + distances

            direction_dotproducts = np.abs(np.matmul(normalsT, line_vector[:, np.newaxis]).flatten())
            angle_diffs = np.abs(np.arccos(direction_dotproducts) - (np.pi / 2))
            valid_angle = np.degrees(angle_diffs) <= 15
            valid_dist = np.abs(dist_diffs) <= 15

            valid = valid_angle & valid_dist
            min_dist_line_i = np.argmin(np.abs(dist_diffs))

            if valid[min_dist_line_i]:
                line_pts[min_dist_line_i].append(pts[:, i])
                if min_dist_line_i != previous_line_i:
                    line_order.append(min_dist_line_i)
                    # line_changes.append((previous_line_i, min_dist_line_i))
                    previous_line_i = min_dist_line_i

    new_lines = []
    # now fit lines
    for line_i, pts in enumerate(line_pts):
        if len(pts) == 0:
            new_lines.append(lines[:, line_i])
            continue

        A = np.concatenate((np.array(pts), np.ones((len(pts), 1))), axis=1)
        _, _, vt = np.linalg.svd(A)
        a, b, c = vt[-1, :]
        norm = math.hypot(a, b)
        if norm > 0:
            a /= norm
            b /= norm
            c /= norm

        if c > 0:
            a *= -1
            b *= -1
            c *= -1
        new_lines.append(np.array([a, b, c]))

    new_lines = einops.rearrange(np.array(new_lines), 'N abc -> abc N', abc=3)
    return new_lines, [np.array(pts).transpose() for pts in line_pts]

def get_appearance_scores(shifted_Hs, init_frame, init_coords, frame, conf):
    do_vis = False
    visualizations = []
    symmetry_resolve_resolution_hw = conf.resolve_symmetry.resolution_hw

    scores = [-np.inf for _ in shifted_Hs]
    H_fit = None
    best_score = -np.inf
    for H_i, shifted_H in enumerate(shifted_Hs):
        if shifted_H is not None:
            shifted_H_fit, H_fit = gu.fit_H_to_size(shifted_H, init_coords,
                                                    symmetry_resolve_resolution_hw,
                                                    fit_margin=0.01)
            if conf.resolve_symmetry.fullfit:
                current_coords = H_warp(shifted_H, init_coords)
                tH, tW = symmetry_resolve_resolution_hw
                target_coords = np.array([[0, tW-1, tW-1, 0],
                                          [0, 0, tH-1, tH-1]]).astype(np.float32)
                H_fit, _ = cv2.findHomography(einops.rearrange(current_coords, 'xy N -> N 1 xy'),
                                              einops.rearrange(target_coords, 'xy N -> N 1 xy'))
                shifted_H_fit = H_compose(shifted_H, H_fit)

            current_corners = H_warp(shifted_H_fit, init_coords)
            current_mask = vu.draw_mask(current_corners, symmetry_resolve_resolution_hw)
            break
    if H_fit is None:
        return scores

    fitted_frame = cv2.warpPerspective(frame, H_fit, symmetry_resolve_resolution_hw[::-1])
    fitted_frame[~current_mask, :] = 0

    if conf.resolve_symmetry.method == 'dino':
        assert symmetry_resolve_resolution_hw == (224, 224)
        dino = conf.resolve_symmetry.extractor
        current_features = dino.extract(fitted_frame, conf.resolve_symmetry.dino_layer)
    
    for H_i, shifted_H in enumerate(shifted_Hs):
        if shifted_H is None:
            continue
        shifted_H_fit = H_compose(shifted_H, H_fit)

        warped_init = cv2.warpPerspective(
            init_frame, shifted_H_fit,
            symmetry_resolve_resolution_hw[::-1])
        warped_init[~current_mask, :] = 0

        if conf.resolve_symmetry.method == 'ssim':
            _, ssim_image = structural_similarity(warped_init, fitted_frame, full=True, channel_axis=2)
            ssim_image = einops.reduce(ssim_image, 'H W bgr -> H W', bgr=3, reduction='mean')
            score = ssim_image[current_mask].mean()
        elif conf.resolve_symmetry.method == 'dino':
            init_features = dino.extract(warped_init, conf.resolve_symmetry.dino_layer)

            similarities = einops.reduce(current_features * init_features, 'H W D -> H W', reduction='sum', D=384)
            score = similarities.mean()  # do not mask by the object mask - different resolution the dino inputs were masked anyway
        else:
            raise ValueError(f"Unknown {conf.resolve_symmetry.method=}")

        if do_vis:
            visualizations.append((warped_init, fitted_frame, score))

        scores[H_i] = score

    scores = np.array(scores)
    if do_vis:
        vis = vu.tile([[init, current] for init, current, score in visualizations])
        vu.imshow("cv: symmetry", vis, keep_history=True)
        # vu.imshow("cv: ssim_vis", vis)
        #     vis = vu.tile([[warped_init, fitted_frame],
        #                    [vu.vis_alignment_plain(warped_init, fitted_frame), ssim_vis]])
        while True:
            c = cv2.waitKey(0)
            if c == ord('q'):
                sys.exit(1)
            elif c == ord(' '):
                break
            elif c == ord('>'):
                raise NextSequence()

    return scores


def recenter_lines(lines, new_center):
    H_center_to_origin = np.eye(3)
    H_center_to_origin[0, 2] = -new_center[0]
    H_center_to_origin[1, 2] = -new_center[1]

    return H_warp_lines(H_center_to_origin, lines), H_center_to_origin

def lines_to_points(lines):
    pts = []
    parents = []
    N_lines = lines.shape[1]
    if N_lines < 2:
        return np.zeros((2, 0), dtype=np.float32)

    for i, line in col_enumerate(lines):
        prev_line = lines[:, i - 1]  # wrap around to the end
        pts.append(gu.line_line_intersection(prev_line, line))
        parents.append(((i-1) % N_lines, i))

    return np.concatenate(pts, axis=1), parents

def line_to_segment(line, pts):
    a, b, c = line

    norm = a**2 + b**2
    assert norm > 0, "Normal vector length must be nonzero"

    
    # only consider the first and the last point. But because the
    # points were found in the affine canonical view, we need to
    # actually sort them, i.e. project onto line direction vector (-b, a)
    line_direction = np.array([[-b, a]]) / norm
    line_pos = np.matmul(line_direction, pts).flatten()
    min_id = np.argmin(line_pos)
    max_id = np.argmax(line_pos)

    endpoints = pts[:, [min_id, max_id]]
    xs = endpoints[0, :]
    ys = endpoints[1, :]

    # formula derivation: {{{
    # The formulas find the point $(x_p, y_p)$ on the line $ax+by+c=0$
    # such that the line segment connecting your original point $(x_0,
    # y_0)$ to $(x_p, y_p)$ is perpendicular to the given line.
    # 
    # Here's a brief derivation:
    # 1.  Let $P_0 = (x_0, y_0)$ be the point to project, and $L: ax+by+c=0$ be the line. The projected point is $P_p = (x_p, y_p)$.
    # 2.  The vector normal to the line $L$ is $\vec{n} = (a, b)$.
    # 3.  The vector $\vec{P_0P_p} = (x_p - x_0, y_p - y_0)$ must be parallel to $\vec{n}$ because the segment $P_0P_p$ is perpendicular to $L$.
    #     So, $(x_p - x_0, y_p - y_0) = k \cdot (a, b)$ for some scalar $k$.
    #     This gives:
    #     $x_p = x_0 + ka$
    #     $y_p = y_0 + kb$
    # 4.  Since $P_p$ is on the line $L$, it satisfies $ax_p + by_p + c = 0$. Substitute the expressions for $x_p, y_p$:
    #     $a(x_0 + ka) + b(y_0 + kb) + c = 0$
    #     $ax_0 + ka^2 + by_0 + kb^2 + c = 0$
    # 5.  Solve for $k$:
    #     $k(a^2 + b^2) = -(ax_0 + by_0 + c)$
    #     $k = -\frac{ax_0 + by_0 + c}{a^2 + b^2}$
    # 6.  Substitute this $k$ back into the expressions for $x_p$ and $y_p$:
    #     $x_p = x_0 - a \frac{ax_0 + by_0 + c}{a^2 + b^2} = \frac{x_0(a^2+b^2) - a(ax_0+by_0+c)}{a^2+b^2} = \frac{b^2x_0 - aby_0 - ac}{a^2+b^2}$
    #     $y_p = y_0 - b \frac{ax_0 + by_0 + c}{a^2 + b^2} = \frac{y_0(a^2+b^2) - b(ax_0+by_0+c)}{a^2+b^2} = \frac{a^2y_0 - abx_0 - bc}{a^2+b^2}$
    # 
    # These are equivalent to the formulas used in the code:
    # $x_p = \frac{b(bx_0 - ay_0) - ac}{a^2+b^2}$
    # $y_p = \frac{a(ay_0 - bx_0) - bc}{a^2+b^2}$

    # }}}
    xs_proj = (b * (b * xs - a * ys) - a * c) / norm
    ys_proj = (a * (-b * xs + a * ys) - b * c) / norm

    endpoints = np.stack((xs_proj, ys_proj), axis=0)
    
    # perpendicular_lines = np.cross(line, gu.e2p(endpoints), axisa=0, axisb=0, axisc=0)
    # intersections = np.cross(perpendicular_lines, line, axisa=0, axisb=0, axisc=0)

    # endpoints = gu.p2e(intersections)
    return endpoints

def line_segment_length(endpoints):
    return np.linalg.norm(endpoints[:, 0] - endpoints[:, 1])

def bruteforce_pointset_matching(pts_a, pts_b):
    all_matchings = []

    if pts_b.size == 0:
        return all_matchings

    N_a = pts_a.shape[1]
    N_b = pts_b.shape[1]

    for N_matches in range(1, min(N_a, N_b) + 1):
        for subset_A_indices in itertools.combinations(range(N_a), N_matches):
            for perm_B_indices in itertools.permutations(range(N_b), N_matches):
                all_matchings.append((subset_A_indices, perm_B_indices))
    # print(f"{len(all_matchings)=}")
    return all_matchings

def find_closest_H(init_coords, current_coords, last_H_init2current, point_matches):
    best_diff = np.inf
    best_H = last_H_init2current.copy()
    best_ids = [[], []]
    most_matches = max((len(a_ids) for a_ids, b_ids in point_matches), default=0)
    for a_ids, b_ids in point_matches:
        if len(a_ids) < most_matches:
            continue
        a_pts = init_coords[:, a_ids]
        b_pts = current_coords[:, b_ids]
        H_init2current = find_transformation_2(a_pts, b_pts, last_H_init2current)

        if gu.good_homography(H_init2current, init_coords):
            H = H_init2current
            H = H / H[2, 2]

            warped_init_coords = H_warp(H, init_coords)
            last_coords = H_warp(last_H_init2current, init_coords)
            diff = np.mean(np.linalg.norm(warped_init_coords - last_coords, axis=0))
            if False and len(a_ids) == 2:
                # print(f"{b_ids=}, {diff=}")
                vis = 255 * np.ones((700, 1400, 3), dtype=np.uint8)
                # print(a_pts)
                # print(b_pts)
                vis = vu.draw_corners(vis, warped_init_coords, color=vu.RED, lineType=cv2.LINE_AA)
                vis = vu.draw_corners(vis, last_coords, color=vu.BLUE, lineType=cv2.LINE_AA)
                for _, pt in col_enumerate(a_pts):
                    vis = vu.circle(vis, H_warp(H, pt), 15, color=vu.RED, thickness=-1)
                    vis = vu.circle(vis, H_warp(last_H_init2current, pt), 15, color=vu.BLUE, thickness=-1)

                for _, pt in col_enumerate(b_pts):
                    vis = vu.circle(vis, pt, 15, color=vu.GREEN, thickness=2)
                vu.imshow("cv: matching", vis, keep_history=True)
                while True:
                    c = cv2.waitKey(5)
                    if c == ord('q'):
                        sys.exit(1)
                    elif c == ord(' '):
                        break
                    elif c == -1:
                        break

            if diff < best_diff:
                best_H = H
                best_diff = diff
                best_ids = (a_ids, b_ids)

    return best_H, best_diff, best_ids

def keep_only_points_close_to_contours(points, mask_contours, max_mask_distance_px=80):
    if points.size == 0:
        return points, np.zeros_like(points) > 0

    flat_contours, _ = einops.pack(mask_contours, 'xy *')
    diff = (einops.rearrange(points, 'xy N_pts -> xy N_pts 1', xy=2) -
            einops.rearrange(flat_contours, 'xy N_contours -> xy 1 N_contours', xy=2))
    dists = einops.reduce(np.square(diff), 'xy N_pts N_contours -> N_pts N_contours', reduction='sum', xy=2)
    min_dist = einops.reduce(dists, 'N_pts N_contours -> N_pts', reduction='min')
    good_mask = min_dist < max_mask_distance_px**2

    return points[:, good_mask], good_mask

def discard_diagonal_matches(matches, parents, N_template_points):
    result = []
    for a_ids, b_ids in matches:
        discard = False
        if len(a_ids) == 2:
            a_consecutive = (abs(a_ids[0] - a_ids[1]) == 1) or (min(a_ids) == 0 and max(a_ids) == N_template_points - 1)
            b_consecutive = len(set.union(*[set(parents[i]) for i in b_ids])) == 3
            if a_consecutive != b_consecutive:
                discard = True

        if not discard:
            result.append((a_ids, b_ids))
    return result
            
def resolve_symmetry_img(best_H, init_coords, best_symmetry_shift_run, init_frame, frame, conf, debug_info):
    symmetry_resolve_shift = 0
    best_H_corners = H_warp(best_H, init_coords)

    shifted_Hs, shifted_coords = gu.get_shifted_homographies(init_coords, best_H_corners)
    scores = get_appearance_scores(shifted_Hs, init_frame, init_coords, frame, conf)
    best_is = np.argsort(scores)[::-1]
    best_i = best_is[0]
    second_best_i = best_is[1]

    best_score = scores[best_i]
    best_score_margin = scores[best_i] - scores[second_best_i]

    if best_i != best_symmetry_shift_run[0]:
        best_symmetry_shift_run[0] = best_i
        best_symmetry_shift_run[1] = 1
    else:
        best_symmetry_shift_run[1] += 1

    # only switch from the closest one if we are quite sure
    _C = conf.resolve_symmetry
    shift_confirmed = False
    if (best_score > _C.abs_thr and best_score_margin > _C.fts_thr):
        debug_info['symmetry_resolve_good_score'] = True
        shift_confirmed = True
    if best_symmetry_shift_run[1] > _C.run_length and best_score > 0:
        debug_info['symmetry_resolve_good_run'] = True
        shift_confirmed = True

    if shift_confirmed:
        best_H = shifted_Hs[best_i]
        symmetry_resolve_shift = best_i

    debug_info['best_symmetry_resolve_H'] = best_H
    debug_info['symmetry_resolve_shift'] = symmetry_resolve_shift
    if debug_info['debug_enabled']:
        msgs = [f"{score:.2f}" for score in scores]
        # from flatsam.utils.io import ansi_text
        # msgs = [ansi_text(msg, bold=i==best_i, underline=i==best_i) for i, msg in enumerate(msgs)]
        print(', '.join(msgs))

    return best_H, symmetry_resolve_shift, shift_confirmed, debug_info

def swap_better_template(best_H_init_to_current, real_init_coords, frames, best_template_frame_idx, best_template_H_to_init):
    undo_info = {'init_coords': real_init_coords.copy(),
                 'init_frame': frames[0]}

    if best_template_frame_idx == 0:
        undo_info['do_nothing'] = True
        return best_H_init_to_current, real_init_coords, frames[0], undo_info

    # best_H_init_to_current: init->current
    # we need to convert to better_template->current
    best_H = H_compose(best_template_H_to_init, best_H_init_to_current)
    H_init_to_best_template = np.linalg.inv(best_template_H_to_init)
    undo_info['H_init_to_best_template'] = H_init_to_best_template
    init_coords = H_warp(H_init_to_best_template, real_init_coords)
    init_frame = frames[best_template_frame_idx]
    return best_H, init_coords, init_frame, undo_info

def undo_swap_better_template(undo_info, best_H_best_template_to_current):
    init_frame = undo_info['init_frame']
    init_coords = undo_info['init_coords']

    if undo_info.get('do_nothing', False):
        best_H_init_to_current = best_H_best_template_to_current
    else:
        best_H_init_to_current = H_compose(undo_info['H_init_to_best_template'], best_H_best_template_to_current)

    return best_H_init_to_current, init_coords, init_frame

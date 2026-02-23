import numpy as np

import flatsam.utils.vis as vu
import flatsam.utils.geom as gu
from flatsam.flatsam import sam_track, flatsam_track
from flatsam.utils.timing import general_time_measurer
from timeit import default_timer as timer

import logging

logger = logging.getLogger(__name__)

def woft_track(sam_predictor, config, frames, init_coords, seq_name, debug=False, debug_fastforward=None):
    from flatsam.woft import WOFT
    tracker = WOFT(config)

    init_frame = frames[0].copy()
    init_mask = vu.draw_mask(init_coords, frames[0].shape[:2]).astype(np.uint8)

    for frame_idx, mask in sam_track(frames, init_mask, sam_predictor, seq_name=seq_name):
        if frame_idx == 0:
            tracker.init(init_frame, init_mask)
            H_cur2init = np.eye(3)
            last_H_cur2init = H_cur2init.copy()
        else:
            try:
                H_cur2init, meta = tracker.track(frames[frame_idx].copy(), debug=debug)
                last_H_cur2init = H_cur2init.copy()
            except Exception:
                logger.exception("WOFT failed")
                H_cur2init = last_H_cur2init.copy()

        vis = None
        debug_info = {'frame_idx': frame_idx, 'debug_enabled': debug,
                      'output_H2init': H_cur2init}
        if frame_idx == 0:
            debug_info['init_coords'] = init_coords

        yield frame_idx, mask, vis, debug_info

def woftsam_track(sam_predictor, config, frames, init_coords, seq_name, debug=False, debug_fastforward=None):
    from flatsam.woftsam import WOFTSAM
    tracker = WOFTSAM(config)

    init_frame = frames[0].copy()
    init_mask = vu.draw_mask(init_coords, frames[0].shape[:2]).astype(np.uint8)

    n_timed_frames = 0
    for frame_idx, mask, vis, debug_info in flatsam_track(sam_predictor, config.flatsam, frames, init_coords, seq_name, debug, debug_fastforward):
        if frame_idx == 0:
            tracker.init(init_frame, init_mask)
            H_cur2init = np.eye(3)
            last_H_cur2init = H_cur2init.copy()

            start_time = timer()
        else:
            n_timed_frames += 1
            try:
                if 'output_H' in debug_info:
                    flatsam_H2init = np.linalg.inv(debug_info['output_H'])
                else:
                    flatsam_H2init = debug_info['output_H2init']
            except Exception:
                logger.exception("Cannot get flatsam H")
                flatsam_H2init = None

            try:
                woftsam_timer = general_time_measurer('WOFTSAM_overall', cuda_sync=False, start_now=True)
                H_cur2init, meta = tracker.track(frames[frame_idx].copy(), robust_H2init=flatsam_H2init, debug=debug)
                woftsam_timer.report()
                last_H_cur2init = H_cur2init.copy()
            except Exception:
                logger.exception("WOFTSAM failed")
                H_cur2init = last_H_cur2init.copy()

        vis = None
        debug_info = {'frame_idx': frame_idx, 'debug_enabled': debug,
                      'output_H2init': H_cur2init}
        if frame_idx == 0:
            debug_info['init_coords'] = init_coords

        yield frame_idx, mask, vis, debug_info

    duration_s = float(timer() - start_time)
    ms_per_frame = (1000 * duration_s) / n_timed_frames
    fps = n_timed_frames / duration_s
    logger.debug(f'WOFTSAM tracking on {mask.shape[1]}x{mask.shape[0]} images: {ms_per_frame:.0f} ms/frame, {fps:0.1f} FPS')


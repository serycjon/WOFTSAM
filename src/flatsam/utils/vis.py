from collections import deque
import cv2
import numpy as np
import einops
import matplotlib.pyplot as plt

from flatsam.utils.io import VideoWriter
import flatsam.utils.geom as gu

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

_IMSHOW_WRITERS = {}
_IMSHOW_HISTORY = deque([], maxlen=300)
_IMSHOW_HISTORY_i = 0

def imshow(win_name, img, scalable=True,
           export_video_path=None,
           export_images=False,
           show_write=False,
           enable_write=True,
           keep_history=False):
    global _IMSHOW_HISTORY_i
    if export_video_path is not None and enable_write:
        if win_name not in _IMSHOW_WRITERS:
            _IMSHOW_WRITERS[win_name] = VideoWriter(export_video_path, images_export=export_images)
        elif str(_IMSHOW_WRITERS[win_name].path) != str(export_video_path):
            _IMSHOW_WRITERS[win_name].close()
            _IMSHOW_WRITERS[win_name] = VideoWriter(export_video_path, images_export=export_images)

        _IMSHOW_WRITERS[win_name].write(img)
    if export_video_path is None or not enable_write or show_write:
        flags = cv2.WINDOW_AUTOSIZE
        if scalable:
            flags = cv2.WINDOW_NORMAL
        cv2.namedWindow(win_name, flags)
        cv2.imshow(win_name, img)
        if keep_history:
            _IMSHOW_HISTORY.append((win_name, img.copy()))
            _IMSHOW_HISTORY_i = len(_IMSHOW_HISTORY) - 1

def imshow_history_show_older():
    global _IMSHOW_HISTORY_i
    _IMSHOW_HISTORY_i -= 1
    if _IMSHOW_HISTORY_i < 0:
        _IMSHOW_HISTORY_i = 0
    win, img = _IMSHOW_HISTORY[_IMSHOW_HISTORY_i]
    cv2.imshow(win, img)

def imshow_history_show_newer():
    global _IMSHOW_HISTORY_i
    _IMSHOW_HISTORY_i += 1
    if _IMSHOW_HISTORY_i >= len(_IMSHOW_HISTORY):
        _IMSHOW_HISTORY_i = len(_IMSHOW_HISTORY) - 1
    win, img = _IMSHOW_HISTORY[_IMSHOW_HISTORY_i]
    cv2.imshow(win, img)

def flush_video_writers():
    for writer in _IMSHOW_WRITERS.values():
        writer.close()


def blend_mask(img, mask, color=(0, 255, 0), alpha=0.5,
               fill=True, contours=True, contour_thickness=1,
               confidence=None):
    '''Blend color mask over image

    img -- 3 channel float32 img with 0-255 values
    mask -- numpy single channel bool img
    color -- three-tuple with 0-255 values RGB
    alpha -- float mask alpha
    contours -- whether to draw mask contours (with alpha=1)
    contour_thickness -- pixel thickness of the contours

    Adapted from:
    https://github.com/karolmajek/Mask_RCNN/blob/master/visualize.py

    '''
    assert mask.dtype == bool
    canvas = img.copy()
    if confidence is not None:
        alpha = alpha * confidence

    if fill:
        color_array = np.array(color)[np.newaxis, np.newaxis, :]
        canvas[mask, :] = canvas[mask] * (1 - alpha) + alpha * color_array

    if contours:
        cnt, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(canvas, cnt, -1, color, contour_thickness)

    return canvas

def draw_mask(corners, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    if not np.all(corners == 0):
        # drawing the all-0 corners results in 1 mask pixel in topleft...
        cv2.fillPoly(mask,
                     [einops.rearrange(
                         np.round(corners).astype(np.int32),
                         'xy N -> N 1 xy', xy=2, N=4)],
                     1)
    return mask > 0


def draw_corners(canvas, corners, color, thickness=2, with_cross=True, with_TL=True,
                 alpha=1, TL_rel_scale=2, TL_radius=None, lineType=None, draw_ori_color=False):
    """ Draw polygon bounded by corners

    args:
        corners: (2, 4) array
    """
    if corners is None:
        return canvas
    assert corners.shape == (2, 4), f"Incorrect corners shape {corners.shape}"
    # pts = np.round(corners).astype(np.int32)
    pts = corners
    pts = einops.rearrange(pts, 'xy N_corners -> N_corners 1 xy', xy=2, N_corners=4)
    vis = canvas.copy()
    vis = polylines(vis, [pts], True, color, thickness, lineType=lineType)
    if draw_ori_color:
        vis = line(vis, tuple(pts[0, 0, :]), tuple(pts[1, 0, :]), draw_ori_color, thickness, lineType=lineType)
    if with_cross:
        vis = line(vis, tuple(pts[0, 0, :]), tuple(pts[2, 0, :]), color, thickness, lineType=lineType)
        vis = line(vis, tuple(pts[1, 0, :]), tuple(pts[3, 0, :]), color, thickness, lineType=lineType)
    if with_TL:
        if TL_radius is None:
            TL_radius = TL_rel_scale * thickness
        vis = cv2.circle(vis, tuple(pts[0, 0, :].astype(np.int32).tolist()), radius=TL_radius, color=color, thickness=-1)

    if alpha != 1:
        vis = cv2.addWeighted(vis, alpha, canvas, (1 - alpha), 0)

    return vis

def line(img, pt1, pt2, color, thickness=None, lineType=None, shift=4):
    """ Same as cv2.line, but does the fractional bit shift inside
    accepts float pt1 and pt2."""
    if shift is not None:
        multiplier = 2**shift
        pt1 = tuple(np.round(multiplier * np.array(pt1)).astype(np.int32).tolist())
        pt2 = tuple(np.round(multiplier * np.array(pt2)).astype(np.int32).tolist())

    return cv2.line(img, pt1, pt2, color, thickness, lineType, shift)


def circle(img, center, radius, color, thickness=None, lineType=None, shift=4):
    """ Same as cv2.circle, but does the fractional bit shift inside
    accepts float center and radius."""
    if shift is not None:
        multiplier = 2**shift
        center = tuple(np.round(multiplier * np.array(center)).astype(np.int32).tolist())
        radius = int(np.round(multiplier * radius).astype(np.int32))

    return cv2.circle(img, center, radius, color, thickness, lineType, shift)


def polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=4):
    """ Same as cv2.polylines, but does the fractional bit shift inside
    accepts float pts."""
    if shift is not None:
        multiplier = 2**shift
        pts = np.round(multiplier * np.array(pts)).astype(np.int32)
    return cv2.polylines(img, pts, isClosed, color, thickness, lineType, shift=shift)

def vis_alignment_plain(src, dst, equalize_hist=False):
    assert src.shape == dst.shape

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if equalize_hist:
        dst_gray = cv2.equalizeHist(dst_gray).astype(np.float32) / 255
        src_gray = cv2.equalizeHist(src_gray).astype(np.float32) / 255
    else:
        dst_gray = dst_gray.astype(np.float32) / 255
        src_gray = src_gray.astype(np.float32) / 255
        dst_gray = (dst_gray - np.amin(dst_gray)) / np.ptp(dst_gray)
        src_gray = (src_gray - np.amin(src_gray)) / np.ptp(src_gray)

    alignment_vis = np.zeros(dst.shape, dtype=np.float32)
    alignment_vis[:, :, 0] = dst_gray
    alignment_vis[:, :, 1] = src_gray
    alignment_vis[:, :, 2] = dst_gray

    alignment_vis = np.uint8(alignment_vis * 255)
    return alignment_vis

def draw_text(img, text, size=3, color=(255, 255, 255),
              pos='bl', thickness=3,
              bg=True, bg_alpha=0.5,
              fit_in=True):
    canvas = img.copy()
    text_margin = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    if isinstance(text, (list, tuple)):
        texts = text
    else:
        texts = [text]

    if isinstance(color, (list, tuple)) and isinstance(color[0], int):
        colors = [color]
        bg_color = (255 - color[0], 255 - color[1], 255 - color[2])
    elif isinstance(color, (list, tuple)) and len(color) == len(texts):
        colors = color
        bg_color = (0, 0, 0)
    else:
        raise ValueError("weird color input")

    while True:
        text_size, text_baseline = cv2.getTextSize(''.join(texts), font, size, thickness)
        text_W, text_H = text_size
        if fit_in:
            if (text_W > canvas.shape[1] or text_H > canvas.shape[0]) and size > 1:
                # try again with smaller size
                size = max(1, size - 1)
                thickness = max(1, thickness - 1)
            else:
                break  # found a fitting size, or cannot shrink any more
        else:
            break
    if pos == 'bl':
        text_bl = (text_margin, canvas.shape[0] - (text_margin + text_baseline))
    elif pos == 'tr':
        text_bl = (canvas.shape[1] - (text_margin + text_W),
                   text_margin + text_H)
    elif pos == 'tl':
        text_bl = (text_margin,
                   text_margin + text_H)
    elif pos == 'br':
        text_bl = (canvas.shape[1] - (text_margin + text_W),
                   canvas.shape[0] - (text_margin + text_baseline))
    elif pos == 'center':
        text_bl = (canvas.shape[1] // 2 - text_W // 2,
                   canvas.shape[0] // 2 + text_H // 2)
    else:
        text_bl = pos

    if bg:
        bg_margin = 2
        bg_canvas = canvas.copy()
        cv2.rectangle(bg_canvas,
                      (text_bl[0] - bg_margin, text_bl[1] + bg_margin + text_baseline),
                      (text_bl[0] + text_W + bg_margin, text_bl[1] - text_H - bg_margin),
                      bg_color, thickness=-1)
        canvas = cv2.addWeighted(bg_canvas, bg_alpha, canvas, (1 - bg_alpha), 0)

    bl = text_bl
    for text_i, text in enumerate(texts):
        c = colors[text_i % len(colors)]
        canvas = cv2.putText(canvas, text,
                             bl, font,
                             size, c, thickness, cv2.LINE_AA)
        bl = (bl[0] + cv2.getTextSize(text, font, size, thickness)[0][0], bl[1])
    return canvas


def blend_colors(a, b, a_alpha):
    """
    args:
        a: 3-tuple of ints (0 - 255) BGR color
        b: 3-tuple of ints (0 - 255) BGR color
        a_alpha: float between 0 and 1
    returns:
        blended: 3-tuple of ints (0 - 255) BGR color
    """

    blended = a_alpha * np.array(a, dtype=np.float64) + (1 - a_alpha) * np.array(b, dtype=np.float64)
    return tuple(np.round(blended).astype(np.int32).tolist())

def place_img_at(img, canvas, tl_row, tl_col):
    H, W = img.shape[:2]
    canvas[tl_row:tl_row + H, tl_col:tl_col + W, :] = img if len(img.shape) == 3 else img[:, :, np.newaxis]


def name_fig(img_list, name_list, size=1, thickness=1, pos='tl'):
    named = []
    for img, name in zip(img_list, name_list):
        named.append(draw_text(img, name, size=size, thickness=thickness, pos=pos))
    return named


def griddify(img_list, cols=None, rows=None):
    N_img = len(img_list)
    if cols is None and rows is None:
        cols = int(np.floor(np.sqrt(N_img)))

    if cols is None:
        cols = int(np.ceil(N_img / rows))

    if rows is None:
        rows = int(np.ceil(N_img / cols))

    grid = []
    for row in range(rows):
        row_imgs = []
        for col in range(cols):
            i = row * cols + col
            img = None
            if i < N_img:
                img = img_list[i]

            row_imgs.append(img)
        grid.append(row_imgs)
    return grid


def tile(img_grid, h_space=1, w_space=None, bg_color=None):
    """Tile images

    args:
        img_grid: a 2D array of images
    """
    if w_space is None:
        w_space = h_space

    rows = len(img_grid)
    cols = len(img_grid[0])

    row_heights = [0] * rows
    col_widths = [0] * cols

    for row_i, row in enumerate(img_grid):
        for col_i, img in enumerate(row):
            if img is None:
                continue
            H, W = img.shape[:2]
            row_heights[row_i] = max(row_heights[row_i], H)
            col_widths[col_i] = max(col_widths[col_i], W)

    out_H = np.sum(row_heights) + (rows - 1) * h_space
    out_W = np.sum(col_widths) + (cols - 1) * w_space
    canvas = np.zeros((out_H, out_W, 3), dtype=img_grid[0][0].dtype)
    if bg_color is not None:
        canvas[:, :] = bg_color

    cur_row = 0
    for row_i, row in enumerate(img_grid):
        cur_col = 0
        for col_i, img in enumerate(row):
            if img is None:
                continue
            place_img_at(img, canvas, cur_row, cur_col)
            cur_col += col_widths[col_i] + w_space

        cur_row += row_heights[row_i] + h_space

    return canvas

def hex_to_bgr(hex_color):
    # Remove the '#' if present
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    # Convert hex to decimal
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    # Return in BGR format
    return (blue, green, red)

def cv2_colormap(img, cmap=None, vmin=None, vmax=None, do_colorbar=False, hatch_params=None,
                 norm=None):
    """ E.g.: vis = colormap(img, plt.cm.viridis) """
    if cmap is None:
        cmap = plt.cm.viridis
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(img)
    if vmax is None:
        vmax = np.nanmax(img)

    if norm is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    elif not norm:
        norm = lambda x: x
    vis = (255 * cmap(norm(img))[..., [2, 1, 0]]).astype(np.uint8)  # RGBA to opencv BGR

    vis[np.isnan(img)] = 0

    if hatch_params is not None:
        vis = cv2_hatch(vis, **hatch_params)

    if do_colorbar:
        vis = cv2_colorbar(vis, vmin, vmax, cmap)

    return vis.copy()

def cv2_hatch(canvas, mask, color=(0, 0, 0), alpha=1, **kwargs):
    """ Put a hatching over the canvas, where mask is True """
    hatching = hatch_pattern(canvas.shape[:2], **kwargs)
    hatch_mask = np.logical_and(mask,
                                hatching > 0)
    hatch_overlay = np.einsum("yx,c->yxc", hatch_mask, color).astype(np.uint8)
    alpha = np.expand_dims(hatch_mask * alpha, axis=2)
    vis = alpha * hatch_overlay + (1 - alpha) * canvas
    return vis.astype(np.uint8)


def hatch_pattern(shape, normal=(2, 1), spacing=10, full=False, **kwargs):
    """ Create a parralel line hatch pattern.  Or a fully filled pattern if 'full' is True.

    Args:
        shape - (H, W) canvas size
        normal - (x, y) line normal vector (doesn't have to be normalized)
        spacing - size of gap between the lines in pixels
        full - use solid fill instead of the pattern

    Outputs:
        canvas - <HxW> np.uint8 image with parallel lines, such that (normal_x, normal_y, c) * (c, r, 1) = 0
    """
    line_type = kwargs.get('line_type', cv2.LINE_8)

    H, W = shape[:2]
    if full:
        canvas = 255 * np.ones((H, W), dtype=np.uint8)
        return canvas

    canvas = np.zeros((H, W), dtype=np.uint8)
    normal = np.array(normal)
    normal = normal / np.sqrt(np.sum(np.square(normal)))

    corners = np.array([[0, 0],
                        [0, H],
                        [W, 0],
                        [W, H]])
    distances = np.einsum("ij,j->i", corners, normal)
    min_c = np.amin(distances)
    max_c = np.amax(distances)
    for c in np.arange(min_c, max_c, spacing):
        res = _img_line_pts((H, W), (normal[0], normal[1], -c))
        if not res:
            continue
        else:
            pt_a, pt_b = res
            cv2.line(canvas,
                     tuple(int(x) for x in pt_a),
                     tuple(int(x) for x in pt_b),
                     255,
                     1,
                     line_type)
    return canvas

def cv2_colorbar(img, vmin, vmax, cmap=None):
    if cmap is None:
        cmap = plt.cm.viridis

    if img.shape[1] < 300:
        scale = int(np.ceil(300 / img.shape[1]))
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cbar_thickness = 20
    separator_sz = 1
    cbar_length = img.shape[1]
    cbar = np.linspace(vmin, vmax, cbar_length, dtype=np.float32)
    cbar = np.tile(cbar, (cbar_thickness, 1))
    cbar = (255 * cmap(norm(cbar))[..., [2, 1, 0]]).astype(np.uint8)  # RGBA to opencv BGR

    separator = np.zeros((separator_sz, cbar.shape[1], cbar.shape[2]), dtype=img.dtype)

    # .copy() to ensure contiguous array? otherwise cv2.putText fails.
    vis = np.vstack((img, separator, cbar)).copy()

    text_margin = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    thickness = 1

    text_min = '{:.2f}'.format(vmin)
    text_min_size, text_min_baseline = cv2.getTextSize(text_min, font, size, thickness)
    text_min_bl = (text_margin,
                   img.shape[0] - (text_margin + text_min_baseline + separator_sz))
    cv2.putText(vis, text_min,
                text_min_bl, font,
                size, (255, 255, 255), thickness, cv2.LINE_AA)

    text_max = '{:.2f}'.format(vmax)
    text_max_size, text_max_baseline = cv2.getTextSize(text_max, font, size, thickness)
    text_max_bl = (img.shape[1] - (text_margin + text_max_size[0]),
                   img.shape[0] - (text_margin + text_max_baseline + separator_sz))
    cv2.putText(vis, text_max,
                text_max_bl, font,
                size, (255, 255, 255), thickness, cv2.LINE_AA)

    return vis.copy()

def _img_line_pts(img_shape, line_eq):
    """ Return boundary points of line in image or False if no exist

    Args:
        img_shape - (H, W) tuple
        line_eq   - 3-tuple (a, b, c) such that ax + by + c = 0

    Returns:
        (x1, y1), (x2, y2) - image boundary intersection points
        or False, if the line doesn't intersect the image
    """
    a, b, c = (float(x) for x in line_eq)
    H, W = img_shape
    if a == 0 and b == 0:
        raise ValueError("Invalid line equation: {}".format(line_eq))
    elif a == 0:
        y = -c / b
        if y < 0 or y >= H:
            return False
        else:
            return (0, y), (W, y)

    elif b == 0:
        x = -c / a
        if x < 0 or x >= W:
            return False
        else:
            return (x, 0), (x, H)
    else:
        pts = set([])

        X_y0_intersection = -c / a
        X_yH_intersection = (-c - b * H) / a

        y0_in = X_y0_intersection >= 0 and X_y0_intersection <= W
        yH_in = X_yH_intersection >= 0 and X_yH_intersection <= W
        if y0_in:
            pts.add((X_y0_intersection, 0))
        if yH_in:
            pts.add((X_yH_intersection, H))

        Y_x0_intersection = -c / b
        Y_xW_intersection = (-c - a * W) / b

        x0_in = Y_x0_intersection >= 0 and Y_x0_intersection <= H
        xW_in = Y_xW_intersection >= 0 and Y_xW_intersection <= H
        if x0_in:
            pts.add((0, Y_x0_intersection))
        if xW_in:
            pts.add((W, Y_xW_intersection))

        if len(pts) == 0:
            return False
        elif len(pts) == 1:
            return False
        elif len(pts) == 2:
            return pts.pop(), pts.pop()
        else:
            raise RuntimeError("Found {} intersections! {}".format(len(pts), pts))

def to_gray_3ch(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR)

def dot(img, cy, cx, color, size=3):
    assert (size % 2) == 1
    deltas = np.arange(size) - size // 2

    cy = int(np.round(cy))
    cx = int(np.round(cx))
    for dy in deltas:
        y = cy + dy
        if y < 0 or y >= img.shape[0]:
            continue
        for dx in deltas:
            x = cx + dx
            if x < 0 or x >= img.shape[1]:
                continue
            img[y, x, ...] = color

def get_aff_patch(img, H, target_sz, laf_scale=3, low_pass=False):
    H_fit = np.array([[1, 0, target_sz // 2],
                      [0, 1, target_sz // 2],
                      [0, 0, 1]], dtype=np.float64)
    H_resize = np.array([[target_sz / (2 * laf_scale), 0, 0],
                         [0, target_sz / (2 * laf_scale), 0],
                         [0, 0, 1]], dtype=np.float64)
    H_extract = gu.H_compose(np.linalg.inv(H), H_resize, H_fit)

    if not low_pass:
        return cv2.warpPerspective(img, H_extract, (target_sz, target_sz)), H_extract

    u, s, vt = np.linalg.svd(H_extract[:2, :2])
    scale = np.mean(np.diag(s))

    # https://dsp.stackexchange.com/questions/75899/appropriate-gaussian-filter-parameters-when-resizing-image
    sigma = 1 / scale

    # Blurring is slow!
    # What about to blur only a small cutout of the object?
    # https://github.com/opencv/opencv/blob/b65fd3b51c692e1bf19fe61c0053967fb6e08a0a/modules/imgproc/src/smooth.dispatch.cpp#L287
    # not exactly the same, but whatever... close enough
    kernel_sz = int(np.round(sigma * 3) * 2 + 1)
    assert (kernel_sz % 2) == 1

    # speed up np.linalg.inv(H_extract)
    # H_extract_inv = np.linalg.inv(H_extract)
    H_extract_inv = gu.H_compose(np.diag(1 / np.diag(H_resize)),
                                 np.array([[1, 0, -1],
                                           [0, 1, -1],
                                           [0, 0, 1]], dtype=np.float64),
                                 H)
    bounds_in_source = gu.H_warp(H_extract_inv, np.array([[0, 0, target_sz - 1, target_sz - 1],
                                                          [0, target_sz - 1, target_sz - 1, 0]], dtype=np.float64))
    crop_bbox = gu.Bbox.from_points(bounds_in_source).rounded_to_int().with_padding(kernel_sz)

    # The overall speedup isn't very big on first 4 videos, but maybe over the whole dataset, it will be worth it...
    if crop_bbox.get_area() < img.shape[1] * img.shape[0]:
        crop = crop_bbox.extract_from_image(img)
        H_crop_shift = np.array([[1, 0, crop_bbox.tl_x],
                                 [0, 1, crop_bbox.tl_y],
                                 [0, 0, 1]], dtype=np.float64)
        H_extract_from_crop = gu.H_compose(H_crop_shift, H_extract)

        filtered = cv2.GaussianBlur(crop, (kernel_sz, kernel_sz), sigma)
        return cv2.warpPerspective(filtered, H_extract_from_crop, (target_sz, target_sz)), H_extract
    else:
        filtered = cv2.GaussianBlur(img, (kernel_sz, kernel_sz), sigma)
        return cv2.warpPerspective(filtered, H_extract, (target_sz, target_sz)), H_extract

def line_eq(img, line_abc, color, thickness=None, lineType=None, shift=4):
    """Draws a line ax + by + c = 0 on the given image."""
    height, width = img.shape[:2]
    a, b, c = line_abc

    points = []

    # 1. Intersect with left boundary (x = 0)
    if b != 0:
        y = -c / b
        if 0 <= y < height:
            points.append((0, y))

    # 2. Intersect with right boundary (x = width - 1)
    if b != 0:
        y = -(a * (width - 1) + c) / b
        if 0 <= y < height:
            points.append((width - 1, y))

    # 3. Intersect with top boundary (y = 0)
    if a != 0:
        x = -c / a
        if 0 <= x < width:
            points.append((x, 0))

    # 4. Intersect with bottom boundary (y = height - 1)
    if a != 0:
        x = -(b * (height - 1) + c) / a
        if 0 <= x < width:
            points.append((x, height - 1))

    # If we have at least two valid points, draw the line
    if len(points) >= 2:
        return line(img, points[0], points[1],
                    color=color, thickness=thickness, lineType=lineType, shift=shift)
    else:
        return img

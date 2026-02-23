import functools
# import logging

import numpy as np
import scipy.linalg
import einops
from shapely import Polygon

class Bbox:
    def __init__(self, tl_x=None, tl_y=None, w=None, h=None):
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.w = w
        self.h = h

        self.br_x = self.tl_x + self.w - 1
        self.br_y = self.tl_y + self.h - 1

    def __repr__(self):
        return f"Bbox(tl_x={self.tl_x}, tl_y={self.tl_y}, w={self.w}, h={self.h})"

    @classmethod
    def from_xyxy(cls, xyxy):
        tl_x = xyxy[0]
        tl_y = xyxy[1]
        br_x = xyxy[2]
        br_y = xyxy[3]
        w = br_x - tl_x + 1
        h = br_y - tl_y + 1

        bbox = cls(tl_x, tl_y, w, h)
        return bbox

    @classmethod
    def from_xywh(cls, xywh):
        bbox = cls(*xywh)
        return bbox

    @classmethod
    def from_cxcywh(cls, cxcywh):
        cx, cy, w, h = cxcywh
        tl_x = cx - (w - 1) / 2
        tl_y = cy - (h - 1) / 2

        bbox = cls(tl_x, tl_y, w, h)
        return bbox

    @classmethod
    def from_mask(cls, binary_image):
        if not np.any(binary_image):
            return Bbox.from_xyxy((0, 0, 0, 0))

        # faster version, thanks to:
        # https://stackoverflow.com/a/31402351/1705970
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        tl_x = cmin
        tl_y = rmin
        br_x = cmax
        br_y = rmax

        bbox = Bbox.from_xyxy((tl_x, tl_y, br_x, br_y))
        return bbox

    @classmethod
    def from_frame(cls, image):
        H, W = image.shape[:2]
        return Bbox.from_xywh((0, 0, W, H))

    @classmethod
    def from_points(cls, pts):
        """Create a bounding box from a bunch of points.

        args:
          pts: (2, N) x, y points
        """
        min_x, max_x = np.amin(pts[0, :]), np.amax(pts[0, :])
        min_y, max_y = np.amin(pts[1, :]), np.amax(pts[1, :])
        return cls.from_xyxy([min_x, min_y, max_x, max_y])

    def as_xyxy(self):
        return [self.tl_x, self.tl_y, self.br_x, self.br_y]

    def as_xywh(self):
        return [self.tl_x, self.tl_y, self.w, self.h]

    def as_points(self):
        return [[self.tl_x, self.tl_y],
                [self.br_x, self.tl_y],
                [self.br_x, self.br_y],
                [self.tl_x, self.br_y]]

    def get_area(self):
        return self.w * self.h

    def get_center(self):
        return [self.tl_x + self.w // 2, self.tl_y + self.h // 2]

    def rounded_to_int(self):
        def round_to_int(x):
            return int(np.round(x))
        return Bbox.from_xyxy((round_to_int(self.tl_x),
                               round_to_int(self.tl_y),
                               round_to_int(self.br_x),
                               round_to_int(self.br_y)))

    def with_padding(self, pad_left, pad_up=None, pad_right=None, pad_down=None):
        if pad_up is None:
            pad_up = pad_left
        if pad_right is None:
            pad_right = pad_left
        if pad_down is None:
            pad_down = pad_up

        return Bbox.from_xyxy((self.tl_x - pad_left,
                               self.tl_y - pad_up,
                               self.br_x + pad_right,
                               self.br_y + pad_down))

    def with_margins(self, margin_fraction):
        return Bbox.from_xyxy((self.tl_x - int(margin_fraction * self.w),
                               self.tl_y - int(margin_fraction * self.h),
                               self.br_x + int(margin_fraction * self.w),
                               self.br_y + int(margin_fraction * self.h)))

    def center_resize(self, wh):
        center = self.get_center()
        tl_x = center[0] - wh[0] // 2
        tl_y = center[1] - wh[1] // 2

        return Bbox.from_xywh((tl_x, tl_y, wh[0], wh[1]))

    def inflate_to_match_aspect_ratio(self, wh_or_bbox, deflate=False):
        if isinstance(wh_or_bbox, Bbox):
            wh = (wh_or_bbox.w, wh_or_bbox.h)
        else:
            wh = wh_or_bbox
        target_ar = wh[0] / wh[1]
        current_ar = self.w / self.h

        scale_x = target_ar / current_ar
        scale_y = current_ar / target_ar

        if not deflate:
            if scale_x >= 1:
                w = self.w * scale_x
                h = self.h
            else:
                w = self.w
                h = self.h * scale_y
        else:
            if scale_x <= 1:
                w = self.w * scale_x
                h = self.h
            else:
                w = self.w
                h = self.h * scale_y

        new_ar = w / h
        assert np.abs(new_ar - target_ar) < 1e-4

        return self.center_resize((w, h))

    def with_margins_min_size(self, min_w, min_h=None):
        if min_h is None:
            min_h = min_w

        missing_w = max(min_w - self.w, 0) / 2
        missing_h = max(min_h - self.h, 0) / 2
        missing_w_margin = missing_w / self.w
        missing_h_margin = missing_h / self.h
        missing_margin = max(missing_w_margin, missing_h_margin)
        if missing_margin > 0:
            bbox = self.with_margins(missing_margin)
        else:
            bbox = self
        return bbox

    def draw(self, canvas, color=(0, 0, 255), thickness=2):
        import cv2
        cv2.rectangle(canvas, (int(self.tl_x), int(self.tl_y)),
                      (int(self.br_x), int(self.br_y)),
                      color, thickness)
        return canvas

    def intersection(self, other):
        intersection_xyxy = [max(self.tl_x, other.tl_x),
                             max(self.tl_y, other.tl_y),
                             min(self.br_x, other.br_x),
                             min(self.br_y, other.br_y)]
        return Bbox.from_xyxy(intersection_xyxy)

    def crop_image(self, img):
        rounded = self.rounded_to_int()
        cropped = img[rounded.tl_y:rounded.br_y,
                      rounded.tl_x:rounded.br_x,
                      ...]
        return cropped

    def extract_from_image(self, image):
        """ Like image[self.tl_y:self.br_y, self.tl_x:self.br_x, ...] but handles out-of-bounds """
        assert all(isinstance(x, int) for x in [self.tl_x, self.tl_y, self.w, self.h])
        img_height, img_width = image.shape[:2]

        # Create the output array filled with zeros (black)
        # Preserve number of channels from original image
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            output = np.zeros((self.h, self.w), dtype=image.dtype)
        else:
            output = np.zeros((self.h, self.w, channels), dtype=image.dtype)

        # Calculate the intersection between the requested rectangle and the image
        # Intersection coordinates relative to the image
        x_start_img = max(0, self.tl_x)
        y_start_img = max(0, self.tl_y)
        x_end_img = min(img_width, self.tl_x + self.w)
        y_end_img = min(img_height, self.tl_y + self.h)

        # Check if the rectangle is completely outside the image
        if x_end_img <= x_start_img or y_end_img <= y_start_img:
            return output  # Return a black rectangle of the specified size

        # Intersection coordinates relative to the output
        x_start_out = max(0, -self.tl_x)
        y_start_out = max(0, -self.tl_y)
        x_end_out = x_start_out + (x_end_img - x_start_img)
        y_end_out = y_start_out + (y_end_img - y_start_img)

        # Copy the overlapping region
        output[y_start_out:y_end_out, x_start_out:x_end_out] = image[y_start_img:y_end_img, x_start_img:x_end_img].copy()

        return output

    def is_pt_inside(self, xy):
        return xy[0] > self.tl_x and xy[0] < self.br_x and \
            xy[1] > self.tl_y and xy[1] < self.tl_y

    def are_pts_inside(self, coords):
        return np.logical_and.reduce((coords[0, :] > self.tl_x,
                                      coords[0, :] < self.br_x,
                                      coords[1, :] > self.tl_y,
                                      coords[1, :] < self.tl_y))

    # def sample_img(self, img, target_box, interpolation=cv2.INTER_NEAREST):
    #     assert target_box.tl_x == 0 and target_box.tl_y == 0

    #     H = H_bbox2bbox(self, target_box)
    #     sample = cv2.warpPerspective(img, H,
    #                                  (target_box.w, target_box.h),
    #                                  flags=interpolation)
    #     return sample

def mask2bbox(mask):
    ''' tl_x, tl_y, w, h '''
    return Bbox.from_mask(mask)

def H_compose(*Hs):
    """ Compose homographies (multiply in reverse order). """
    for H in Hs:
        if H is None:
            return None
    result = functools.reduce(np.dot, reversed(Hs))
    result /= result[2, 2]  # normalize to 1 in bottomright

    return result


def e2p(xs):
    ''' converts (D, N) euclidean coordinates to (D+1, N) projective (homogenous coords) '''
    return np.vstack((xs, np.ones(xs.shape[1])))


def p2e(xs):
    ''' converts (D+1, N) homogenous coordinates to (D, N) euclidean '''
    if xs.size == 0:
        return xs
    return xs[:-1, :] / xs[-1, np.newaxis, :]

def H_warp(H, pts):
    '''
    args:
        pts: (D, N) D-dimensional euclidean coordinates
    returns:
        out: (D, N) D-dimensional euclidean coordinates (≈ H*pts)
    '''
    assert H.shape == (3, 3)
    flat_input = False
    if len(pts.shape) == 1:
        pts = einops.rearrange(pts, 'coords -> coords 1')
        flat_input = True

    result = p2e(np.matmul(H, e2p(pts)))

    if flat_input:
        result = result.flatten()

    return result

def H_warp_lines(H, lines):
    '''
    args:
        lines: (D, N) D-dimensional homogeneous line coordinates (a, b, c from ax + by + c = 0)
    returns:
        out: (D, N) D-dimensional homogeneous line coordinates (≈ H^{-T}*lines)
    '''
    assert H.shape == (3, 3)
    flat_input = False
    if len(lines.shape) == 1:
        lines = einops.rearrange(lines, 'coords -> coords 1')
        flat_input = True

    result = np.matmul(np.linalg.inv(H).transpose(), lines)

    if flat_input:
        result = result.flatten()

    return result

def H_bbox2bbox(src, dst):
    ''' Compute homography mapping bounding boxes

    Args:
        src: Bbox
        dst: Bbox
    '''
    # homography contstructed by unshifting src box topleft to zero,
    # scaling to the dst box size and shifting to the crop box topleft
    H_unshift = np.eye(3)
    H_unshift[0, 2] = -src.tl_x  # unshift x
    H_unshift[1, 2] = -src.tl_y  # unshift y

    scale_h = dst.h / float(src.h)
    scale_w = dst.w / float(src.w)
    H_scale = np.diag((scale_w, scale_h, 1))

    H_shift = np.eye(3)
    H_shift[0, 2] = dst.tl_x  # shift x
    H_shift[1, 2] = dst.tl_y  # shift y

    # H operates on data from left -> reverse order
    H = np.matmul(H_shift, np.matmul(H_scale, H_unshift))
    H /= H[2, 2]
    return H

class EmptyMaskException(RuntimeError):
    pass

def fit_H_to_size(H, src_points, target_hw, fit_margin, keep_aspect_ratio=True):
    dst_points = H_warp(H, src_points)
    dst_bbox = Bbox.from_points(dst_points).with_margins(fit_margin)
    target_box = Bbox.from_xywh((0, 0, target_hw[1], target_hw[0]))

    # keep aspect ratio of the original image - crop with the same aspect ratio as the target box
    dst_bbox = dst_bbox.inflate_to_match_aspect_ratio((target_hw[1], target_hw[0]))
    H_fit = H_bbox2bbox(dst_bbox, target_box)

    H_fitted = H_compose(H, H_fit).astype(np.float32).copy()
    return H_fitted, H_fit

def find_TRS(src_coords, dst_coords):
    """Estimate translation, rotation, scale transformation mapping from src to dst coordinates
    The TRS transformation is estimated by least squares.

    Rotation matrix: [[cos, -sin], [sin, cos]]
    Scaled rotation: [[A, B], [-B, A]]
    TRS: [[A, B, C], [-B, A, D]]

    TRS * [X, Y, 1]^T = [[A*X + B*Y    + C*1 + D*0],
                         [A*Y + B*(-X) + C*0 + D*1]]

    Each correspondence thus adds into the system:
    [X,  Y, 1, 0]                     [X']
    [Y, -X, 0, 1]  * [A, B, C, D]^T = [Y']

    args:
        src_coords: (xy, N) numpy array
        dst_coords: (xy, N) numpy array
    returns:
        TRS: (3, 3) numpy array with the TRS transformation matrix
    """
    X, Y = src_coords[0, :], src_coords[1, :]
    X_prime, Y_prime = dst_coords[0, :], dst_coords[1, :]
    ones, zeros = np.ones_like(X), np.zeros_like(X)

    ax = np.stack((X,  Y, ones,  zeros), axis=1)
    ay = np.stack([Y, -X, zeros, ones], axis=1)

    A = np.concatenate((ax, ay), axis=1)  # each correspondence crammed into one row, so that is easy to interleave next
    b = np.stack((X_prime, Y_prime), axis=1)

    # interleave the two parts, such that we get the standard form (each correspondence gives two consecutive rows in A)
    A = einops.rearrange(A, 'N (two four) -> (N two) four', two=2, four=4)  # add batch
    b = einops.rearrange(b, 'N (two one) -> (N two) one', two=2, one=1)  # add batch

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    A, B, C, D = x.flatten()

    TRS = np.array([[A, B, C],
                    [-B, A, D],
                    [0, 0, 1]])

    return TRS

def find_T(src_coords, dst_coords):
    """Estimate translation mapping from src to dst coordinates (by least squares)

    args:
        src_coords: (xy, N) numpy array
        dst_coords: (xy, N) numpy array
    returns:
        T: (3, 3) numpy array with the translation in a homography form
    """
    shift = einops.reduce(dst_coords - src_coords, 'xy N -> xy', reduction='mean', xy=2)

    T = np.array([[1, 0, shift[0]],
                  [0, 1, shift[1]],
                  [0, 0, 1]])
    return T
def get_shifted_homographies(src_coords, dst_coords):
    assert src_coords.shape == dst_coords.shape
    import cv2
    N_pts = dst_coords.shape[1]
    result_Hs = [None for _ in range(N_pts)]
    result_coords = [None for _ in range(N_pts)]
    for shift in range(N_pts):
        shifted_dst_coords = np.roll(dst_coords, shift, axis=1)
        shifted_H, _ = cv2.findHomography(einops.rearrange(src_coords, 'xy N -> N 1 xy', xy=2),
                                          einops.rearrange(shifted_dst_coords, 'xy N -> N 1 xy', xy=2))
        if shifted_H is None or np.linalg.matrix_rank(shifted_H) != 3:
            shifted_H = None

        result_Hs[shift] = shifted_H
        result_coords[shift] = shifted_dst_coords
    return result_Hs, result_coords

def line_line_intersection(line_1, line_2):
    """
    args:
        line_1: (3, 1) array with line equation (Ax + By + C = 0)
        line_2: (3, 1) array with line equation (Ax + By + C = 0)
    returns:
        pt: (3, 1) array with homogeneous coordinates of the intersection
    """

    a1, b1, c1 = line_1.flatten()
    a2, b2, c2 = line_2.flatten()

    pt = np.array([[b1 * c2 - b2 * c1],
                   [a2 * c1 - a1 * c2],
                   [a1 * b2 - a2 * b1]])
    return pt

def polygon_area(pts, restrict_to_hw=None):
    poly = Polygon(einops.rearrange(pts, 'xy N -> N xy', xy=2))
    is_fully_inside = None

    if restrict_to_hw:
        H, W = restrict_to_hw
        frame_poly = Polygon([[0, 0],
                              [W, 0],
                              [W, H],
                              [0, H]])

        inside_poly = poly.intersection(frame_poly)
        is_fully_inside = inside_poly.area == poly.area
        poly = inside_poly

    return poly.area, is_fully_inside

def convex_quad(pts):
    signs = []
    for i in range(pts.shape[1]):
        a = pts[:, i] - pts[:, (i - 1) % 4]
        b = pts[:, (i + 1) % 4] - pts[:, i]
        z = a[0] * b[1] - a[1] * b[0]  # signed area
        signs.append(z > 0)

    return np.all(signs) or not np.any(signs)

def twisting_homography(H, pts):
    """
    args:
        H: numpy (3, 3) array
        pts: numpy (xy, N) array of points
    returns:
        True if the homography "twists" the points
    """
    projected = H_warp(H, pts)
    return not convex_quad(projected)

def signed_area(pts):
    x, y = pts
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    return 0.5 * np.sum(x * y_next - x_next * y)

def flipped_homography(H, pts):
    return np.sign(signed_area(pts)) != np.sign(signed_area(H_warp(H, pts)))

def good_homography(H, init_coords):
    return H is not None and np.linalg.matrix_rank(H) == 3 and not twisting_homography(H, init_coords) and not flipped_homography(H, init_coords)

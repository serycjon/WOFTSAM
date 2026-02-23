import torch
import einops

def torch_get_featuremap_coords(feature_map, device=None,
                                keep_shape=False):
    """ get coordinate map corresponding to a feature map

    args:
        feature_map: (..., H, W) tensor
        keep_shape: boolean (default False). Setting it to True does not flatten the output coordinates

    returns:
        xy: (2, H*W) tensor with x, y coordinates.  (2, H, W) tensor if keep_shape == True
    """
    if type(feature_map) is tuple and len(feature_map) == 2:
        H, W = feature_map
        assert device is not None
    else:
        H, W = feature_map.shape[-2:]
        if device is None:
            device = feature_map.device
    xy = unravel_indices(torch.arange(H * W, device=device), (H, W), stack_dim=0)

    if keep_shape:
        xy = einops.rearrange(xy, 'xy (H W) -> xy H W', H=H, W=W, xy=2)
    return xy

def unravel_indices(indices, shape, stack_dim=-1, np_order=False):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).

    From https://github.com/pytorch/pytorch/issues/35674#issuecomment-739560051
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')

    if np_order:  # row, column (y, x)
        coord = torch.stack(coord[::-1], dim=stack_dim)
    else:  # column, row (x, y)
        coord = torch.stack(coord, dim=stack_dim)
    assert coord.device == indices.device

    return coord

def remap(x, src_low, src_high, dst_low, dst_high):
    return ((x - src_low) / (src_high - src_low)) * (dst_high - dst_low) + dst_low

def dim_enumerate(xs, dim, keepdim=False):
    try:
        N_elems = xs.shape[dim]
    except Exception:
        return None

    slices = [slice(None) for _ in xs.shape]  # slice(None) is like ":"
    for i in range(N_elems):
        if keepdim:
            slices[dim] = slice(i, i + 1)
        else:
            slices[dim] = i
        yield (i, xs[tuple(slices)])

def col_enumerate(xs, keepdim=False):
    yield from dim_enumerate(xs, dim=1, keepdim=keepdim)


def get_featuremap_coords(feature_map):
    """ get coordinate map corresponding to a feature map

    args:
        feature_map: (..., H, W) np array or tensor, or tuple (H, W)

    returns:
        xy: (2, H*W) np array or tensor with x, y coordinates
    """
    if type(feature_map) is tuple and len(feature_map) == 2:
        H, W = feature_map
    else:
        H, W = feature_map.shape[-2:]
    ys, xs = np.unravel_index(np.arange(H * W), (H, W))
    xy = np.stack((xs, ys), axis=0)
    return xy

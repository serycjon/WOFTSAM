from pathlib import Path

import numpy as np
import einops

def pot_sequences(base_dir, split=None):
    assert split is None
    all_names = [path.stem for path in sorted((base_dir / 'extracted' / 'images').glob('*'))]
    POT_210_names = [seq_name for seq_name in all_names
                     if int(seq_name.split('_')[0][1:]) <= 30]
    return POT_210_names

def pt_sequences(base_dir, split=None):
    if split is not None:
        try:
            split_file = dict(train='training_set.txt', test='test_seq.txt')[split]
            seq_list_path = base_dir / 'training-test-splits' / split_file
        except KeyError:
            seq_list_path = Path(split)
            assert seq_list_path.exists(), "Unknown PlanarTrack split. Should be 'train', 'test', or a path to textfile."

        with seq_list_path.open('r') as fin:
            seq_names = [name.strip() for name in fin.readlines()]
        return seq_names
    else:
        return [path.stem for path in sorted((base_dir / 'sequences').glob('*'))]

def poic_sequences(base_dir, split=None):
    assert split is None
    return [path.stem for path in sorted((base_dir / 'sequences').glob('*'))]

def pot_pos_gt(base_dir, seq_name):
    annot = einops.rearrange(np.loadtxt(base_dir / 'annotation' / 'annotation' / f'{seq_name}_gt_points.txt'),
                           'frames (N xy) -> frames xy N', xy=2, N=4)
    flags = np.loadtxt(base_dir / 'annotation' / 'annotation' / f'{seq_name}_flag.txt')
    valid = np.any(einops.rearrange(annot, 'frames N xy -> frames (N xy)') != 0, axis=1) & (flags == 0)
    return annot, valid
    
def pt_pos_gt(base_dir, seq_name):
    annot = einops.rearrange(np.loadtxt(base_dir / 'annos' / f'{seq_name}.txt'),
                            'frames (N xy) -> frames xy N', xy=2, N=4)
    flags = np.loadtxt(base_dir / 'annos' / f'{seq_name}_flag.txt')
    valid = np.any(einops.rearrange(annot, 'frames N xy -> frames (N xy)') != 0, axis=1) & (flags == 0)
    return annot, valid
    
def poic_pos_gt(base_dir, seq_name):
    # first line is a header
    # first column is a frame name
    annot = einops.rearrange(np.loadtxt(base_dir / 'gt' / f'{seq_name}.txt', skiprows=1, usecols=(1,2,3,4,5,6,7,8)),
                            'frames (N xy) -> frames xy N', xy=2, N=4)
    valid = np.any(einops.rearrange(annot, 'frames N xy -> frames (N xy)') != 0, axis=1)
    return annot, valid

def pot_img_paths(base_dir, seq_name):
    return list(sorted((base_dir / 'extracted' / 'images' / seq_name).glob('*')))

def pt_img_paths(base_dir, seq_name):
    return list(sorted((base_dir / 'sequences' / seq_name).glob('*')))

def poic_img_paths(base_dir, seq_name):
    return list(sorted([path for path in (base_dir / 'sequences' / seq_name).glob('*')
                        if path.is_file() and path.suffix != '.txt']))

_pt_att_names = ['occlusion', 'blur', 'rotation', 'scale', 'perspective', 'out-of-view', 'low-resolution', 'bg-clutter']

def pt_attributes(base_dir, seq_name):
    # att_name = ['Occlusion', 'Motion Blur', 'Rotation', 'Scale Variation',
    #             'Perspective Distortion', 'Out-of-View', 'Low Resolution',
    #             'Background Clutter'];

    attr_mask = np.loadtxt(base_dir / 'challenging_factor' / f'{seq_name}_challenging_factor.txt', delimiter=',') > 0
    attributes = np.array(_pt_att_names)[attr_mask].tolist()
    return attributes
    
default_base_dirs = dict(pt=Path("/mnt/datasets/PlanarTrack/PlanarTrack-Data/"),
                         pot=Path("/mnt/datasets/POT-210/"),
                         poic=Path("/mnt/datasets/POIC/"))
load_gt_funs = dict(pt=pt_pos_gt, pot=pot_pos_gt, poic=poic_pos_gt)
get_sequences = dict(pt=pt_sequences, pot=pot_sequences, poic=poic_sequences)
get_img_paths = dict(pt=pt_img_paths, pot=pot_img_paths, poic=poic_img_paths)

def load_results(base_dir, seq_name, suffix, seqname_subdir):
    if seqname_subdir:
        src_dir = base_dir / seq_name
    else:
        src_dir = base_dir

    if suffix is None:
        result_candidates = list(src_dir.glob(f'{seq_name}*.txt'))
        assert len(result_candidates) == 1
        result_path = result_candidates[0]
    else:
        result_path = src_dir / f'{seq_name}{suffix}.txt'
    return einops.rearrange(np.loadtxt(result_path), 'frames (N xy) -> frames xy N', N=4, xy=2)

def alignment_error(gt, pred):
    return np.sqrt(einops.reduce(np.square(gt - pred), 'frames xy N -> frames', reduction='sum', N=4, xy=2) / 4)

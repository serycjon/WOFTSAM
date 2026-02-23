# Accurate Planar Tracking With Robust Re-Detection

[Project Page](https://cmp.felk.cvut.cz/~serycjon/WOFTSAM/)

Official implementation of:

```
Accurate Planar Tracking With Robust Re-Detection
Serych J., Matas J.
2026
```

**This repository provides the code for SAM-H and WOFTSAM achieving a state-of-the-art performance on POT-210 and PlanarTrack planar object tracking benchmarks.**

[![Demo GIF](compilation_grid.gif "click to watch full video")](https://cmp.felk.cvut.cz/~serycjon/WOFTSAM/visuals/compilation_grid.mp4)

## Installation
```bash
# clone this repository
git clone https://github.com/serycjon/woftsam.git
cd woftsam

# create and activate a virtualenv
python -m venv venv
source venv/bin/activate

# install
pip install -e .
```

Download the [SAM 2.1 Hiera tiny model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) and put it into `checkpoints` directory.

```bash
mkdir checkpoints && cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt && cd ..
```

This is tested to work with Python 3.11, CUDA 12.4.0. For future reference, the exact current versions of the dependencies are listed in `dependencies.txt`.

## Quick WOFTSAM Demo
Running `demo.py` should show a progress bar and create a `demo_out.mp4` video in about two and half minutes.
```bash
python demo.py
```

## PlanarTrack Re-Annotations

The PlanarTrack initial frame re-annotations are provided in the [PlanarTrack_init_reannotation](PlanarTrack_init_reannotation/) directory in the standard planar tracking format - four corner coordinates stored as `x1 y1 x2 y2 x3 y3 x4 y4`.

## Evaluation
### Dataset Preparation
First, you will need to get the POT-210 and the PlanarTrack datasets.

[POT-210](https://web.archive.org/web/20250419014540/https://www3.cs.stonybrook.edu/~hling/data/POT-210/planar_benchmark.html) is expected in the following directory structure:
```
POT-210
 - annotation
   - annotation
     - V01_1_gt_points.txt
	 - V01_1_flag.txt
	 - V01_2_gt_points.txt
	 - V02_2_flag.txt
	 - ...
   - WOFT_reannotation_coarse_onlychanged
     - V01_1_gt_points.txt
	 - V01_1_flag.txt
	 - V01_2_gt_points.txt
	 - V02_2_flag.txt
	 - ...
 - extracted
   - images
     - V01_1
	   - 00000001.jpg
	   - 00000002.jpg
	   ...
	   - 00000501.jpg
	 - V01_2
	   - ...
```

The original website is down at the moment, but the data can be downloaded from the [website](https://liangp.github.io/data/pot280/) of the extended POT-280 dataset. Dataset videos were converted to images using [this script](extract_pot_frames.sh).

The `WOFT_reannotation_coarse_onlychanged` can be downloaded [here](https://cmp.felk.cvut.cz/~serycjon/WOFT/results/POT-210-reannot_coarse_onlychanged.tar.gz).

[PlanarTrack](https://hengfan2010.github.io/projects/PlanarTrack/) is expected in the following directory structure:
```
PlanarTrack
  - annos
    - Seq_00001.txt
	- Seq_00001_flag.txt
    - Seq_00002.txt
	- Seq_00002_flag.txt
	- ...
  - sequences
    - Seq_00001
	  - 00001.jpg
	  - 00002.jpg
	  - ...
    - Seq_00002
	  - ...
  - training-test-splits
    - test_seq.txt
```

For easy working with the datasets, rewrite `default_base_dirs` in `src/flatsam/utils/evaluation.py`:
```python
default_base_dirs = dict(pt=Path("/path/to/your/PlanarTrack/"),
                         pot=Path("/path/to/your/POT-210/"))
```
Otherwise the dataset base directory must be directly specified when running the evaluation scripts.

### Tracking
Execute the following to run the trackers

```bash
# PlanarTrack
python experiments/pt_sam_run.py --split test /path/to/output/directory/ -c configs/woftsam.py
python experiments/pt_sam_run.py --dataset pot /path/to/output/directory/ -c configs/woftsam.py

# POT-210
python experiments/pt_sam_run.py --split test /path/to/output/directory/ -c configs/sam_h.py
python experiments/pt_sam_run.py --dataset pot /path/to/output/directory/ -c configs/sam_h.py
```

The script creates subdirectory structure in the `/path/to/output/directory/` based on the config name and the dataset.

You can force the method to run on a given CUDA GPU with e.g. `--gpu 2` (or other 0-based index).

For PlanarTrack, run with the [reannotated GT initialization](#planartrack-re-annotations) by adding `--initfix PlanarTrack_init_reannotation/`.

In case you didn't configure the default dataset base dirs ([see previous section](#dataset-preparation)), you must specify the path to the relevant dataset with `--dataset_base /the/path/to/the/dataset/`.

### Evaluation
The `pt_sam_run.py` script puts the outputs in the standard planar tracking format at `/path/to/output/directory/[config_name]/masks/[sequence_name]/[sequence_name]_FLATSAM.txt`. Each line of that file represents the 4 control points coordinates on one frame of the video in an `x1 y1 x2 y2 x3 y3 x4 y4` format.

To get the P@5 and P@15 scores on PlanarTrack, run:
```bash
python experiments/pt_eval.py /path/to/output/directory/[config_name]/{masks,flatsam_test_ae.pkl} --split test

# and for the reannotation results:
python experiments/pt_eval.py /path/to/output/directory/initfix_[config_name]/{masks,flatsam_test_ae.pkl} --split test
```

(`{masks,flatsam_test_ae.pkl}` is bash brace expansion. It is just a shortcut for writing `python experiments/pt_eval.py /path/to/output/directory/[config_name]/masks /path/to/output/directory/[config_name]/flatsam_test_ae.pkl --split test`)


To get the P@5 and P@15 scores on POT-210, run:
```bash
python experiments/pt_eval.py /path/to/output/directory/pot_[config_name]/{masks,flatsam_test_ae.pkl} --reannot
```

## License
This work is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The `src/flatsam/woft_stuff/raft_core` directory contains a slightly modified copy of [RAFT](https://github.com/princeton-vl/RAFT), which is licensed under BSD-3-Clause license, except for the `src/flatsam/woft_stuff/raft_core/weighted_raft.py`, which is again licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The `demo` directory includes a sample from the POT-210 dataset.

## Citing WOFTSAM
If you use WOFTSAM, SAM-H or the PlanarTrack initial frame reannotations in your research, please cite our paper with the following BibTeX:

```bibtex
@article{serych2026woftsam,
  title={Accurate Planar Tracking With Robust Re-Detection},
  author={Serych, Jonas and Matas, Jiri},
  journal={arXiv preprint arXiv:2602.19624},
  year={2026}
}
```

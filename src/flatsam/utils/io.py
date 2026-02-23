from contextlib import contextmanager
import tempfile
from pathlib import Path
import logging
from functools import wraps
import traceback
import socket
import argparse

import cv2

logger = logging.getLogger(__name__)

@contextmanager
def ram_directory_of_images(images, seq_name=None, double_first_frame=False, first_n=None):
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp_directory:
        directory = Path(tmp_directory)
        if seq_name is not None:
            directory = directory / seq_name
            directory.mkdir(parents=True, exist_ok=True)

        logger.debug("Writing images to RAM-disk for SAM2")
        for img_i, img in enumerate(images):
            if first_n is not None and img_i >= first_n:
                break
            if double_first_frame and img_i == 0:
                cv2.imwrite(str(directory / f'{img_i:05d}.jpg'), img)
            cv2.imwrite(str(directory / f'{img_i + 1:05d}.jpg'), img)
        logger.debug("DONE writing images to RAM-disk for SAM2")

        try:
            yield directory
        finally:
            # cleanup
            pass

class VideoWriter():
    def __init__(self, path, fps=30, images_export=False, ext='jpg'):
        path = Path(path)
        self.do_write = path is not None
        if self.do_write:
            if images_export:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.path = path
        self.fps = fps
        self.images_export = images_export
        self.frame_i = 0
        self.ext = ext

    def write(self, frame, frame_name=None):
        if self.writer is None and self.do_write and not self.images_export:
            codec = 'mp4v'
            # codec = 'avc1'
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), self.fps,
                                          (frame.shape[1], frame.shape[0]))

        if self.do_write:
            if self.images_export:
                if frame_name is not None:
                    out_path = self.path / f'{frame_name}.{self.ext}'
                else:
                    out_path = self.path / f'{self.frame_i:08d}.{self.ext}'

                cv2.imwrite(str(out_path), frame)
                self.frame_i += 1
            else:
                self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()

    def __del__(self):
        self.close()
        if self.do_write:
            print(f'Exported video to {self.path}')

class GeneralVideoCapture(object):
    """A cv2.VideoCapture replacement, that can also read images in a directory"""

    def __init__(self, path, reverse=False):
        images = Path(path).is_dir()
        self.image_inputs = images
        if images:
            self.path = Path(path)
            self.images = sorted([f.name for f in self.path.glob('*') if f.suffix in ['.jpg', '.png', '.jpeg']])
            if len(self.images) == 0:
                print(self.path, list(self.path.glob('*')))
            if reverse:
                self.images = self.images[::-1]
            self.i = 0
        else:
            self.cap = cv2.VideoCapture(str(path))

    def read(self):
        if self.image_inputs:
            if self.i >= len(self.images):
                return False, None
            img_path = self.path / self.images[self.i]
            self.frame_src = self.images[self.i]
            img = cv2.imread(str(img_path))
            self.i += 1
            return True, img
        else:
            return self.cap.read()

    def release(self):
        if self.image_inputs:
            return None
        else:
            return self.cap.release()

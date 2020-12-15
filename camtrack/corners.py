#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:
    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class CornerDetector:
    def __init__(self, width, height):
        self.pyr = None
        self.levels = None
        self.last_id = 0
        self.min_dist = 10
        self.blk_size = 7
        self.win_size = (10, 10)
        self.corners = 10000

        self.corner_pts = np.ndarray((0, 2), dtype=np.float32)
        self.corner_ids = np.array([], dtype=np.int32)
        self.corner_szs = np.array([], dtype=np.float32)
    
    def next_frame(self, image):
        image = (image * 255).astype(np.uint8)
        pyr = self._pyr(image)

        last_pyr = self.pyr
        self.pyr = pyr

        if last_pyr is not None:
            self._track(last_pyr, pyr)

        self._detect(pyr)

        return FrameCorners(self.corner_ids, self.corner_pts, self.corner_szs)

    def _pyr(self, im):
        levels, pyr = cv2.buildOpticalFlowPyramid(im, self.win_size, 3, None, False)

        if self.levels is None:
            self.levels = levels
        else:
            assert self.levels == levels

        return [i.astype(np.uint8) for i in pyr]

    def _detect(self, pyr):
        mask = np.full(pyr[0].shape, 255, dtype=np.uint8)
        for x, y, sz in np.column_stack((self.corner_pts, self.corner_szs)):
            mask = cv2.circle(mask, (np.int(x), np.int(y)), np.int(sz), 0, -1)

        n = self.corner_szs.size

        if n >= self.corners:
            return

        pts = []
        ids = []
        szs = []

        for i in range(self.levels):
            if i != 0:
                mask = cv2.pyrDown(mask).astype(np.uint8)

            feat = cv2.goodFeaturesToTrack(pyr[i],
                    maxCorners=self.corners - n,
                    qualityLevel=0.06,
                    minDistance=self.min_dist,
                    mask=mask,
                    blockSize=self.blk_size)

            if feat is None: # why does this happen???
                continue

            factor = 2.0 ** i

            for x, y in feat.reshape(-1, 2):
                if not mask[np.int(y), np.int(x)]:
                    continue

                pts.append((x * factor, y * factor))
                ids.append(self.last_id)
                szs.append(self.blk_size * factor)
                self.last_id += 1

                mask = cv2.circle(mask, (np.int(x), np.int(y)), self.blk_size, 0, -1)

        if pts:
            self.corner_pts = np.concatenate((self.corner_pts, pts))
            self.corner_ids = np.concatenate((self.corner_ids, ids))
            self.corner_szs = np.concatenate((self.corner_szs, szs))

    def _track(self, pyr0, pyr1):
        if self.corner_pts.size == 0:
            return

        pts0 = self.corner_pts
        pts1, status, err = cv2.calcOpticalFlowPyrLK(
            pyr0[0], pyr1[0], np.asarray(pts0, dtype=np.float32), None,
            winSize=self.win_size,
            minEigThreshold=1e-3
        )

        tracked = status.flatten() == 1
        self.corner_pts = pts1[tracked]
        self.corner_ids = self.corner_ids[tracked]
        self.corner_szs = self.corner_szs[tracked]


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    detector = CornerDetector(*frame_sequence.frame_shape[:2])

    for frame, image in enumerate(frame_sequence):
        builder.set_corners_at_frame(frame, detector.next_frame(image))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

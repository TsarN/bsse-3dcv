#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import attr
import click

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *

REPROJ_ERR = 2.0

@attr.s
class CamTrack:
    corner_storage           = attr.ib()
    known_views              = attr.ib()
    point_cloud_builder      = attr.ib()
    intrinsic_mat            = attr.ib()

    @property
    def frame_count(self):
        return len(self.corner_storage)

    @property
    def initial_frames(self):
        return [self.known_views[i][0] for i in range(2)]

    def run(self):
        self.known_views = list(self.known_views)
        self.init_known_views()

        self.view_mats = [pose_to_view_mat3x4(self.known_views[0][1])] * self.frame_count
        self.view_mats[self.initial_frames[1]] = pose_to_view_mat3x4(self.known_views[1][1])
        self.inlier_counts = [-1] * self.frame_count
        initial_view_mats = [self.view_mats[self.initial_frames[i]] for i in range(2)]
        camera_centers = [to_camera_center(initial_view_mats[i]) for i in range(2)]
        self.initial_distance = np.linalg.norm(camera_centers[0] - camera_centers[1])
        self.solved_frames = np.zeros(self.frame_count).astype(bool)
        self.solved_frames[self.initial_frames[0]] = True
        self.solved_frames[self.initial_frames[1]] = True

        self.triangulate_initial()

        fst, snd = sorted(self.initial_frames)

        last = -1
        while last != np.count_nonzero(self.solved_frames):
            last = np.count_nonzero(self.solved_frames)

            for i in range(self.frame_count):
                self.track_catch(i)

            for i in reversed(range(self.frame_count)):
                self.track_catch(i)

        if last != self.frame_count:
            raise RuntimeError("Couldn't solve for every frame")

    def track_catch(self, frame, skip=False):
        if frame in self.initial_frames:
            return
        try:
            self.track(frame)
        except KeyboardInterrupt:
            raise
        except SyntaxError:
            raise
        except:
            return
        self.solved_frames[frame] = True
        print(f"Solved frame #{frame} (solved {np.count_nonzero(self.solved_frames)} out of {self.frame_count})")
        if not skip and np.count_nonzero(self.solved_frames) % 5 == 0:
            for i in range(self.frame_count):
                self.track_catch(i, True)

            for i in reversed(range(self.frame_count)):
                self.track_catch(i, True)

    def triangulate_initial(self):
        frame_corners = [self.corner_storage[self.initial_frames[i]] for i in range(2)]
        frame_view_mats = [self.view_mats[self.initial_frames[i]] for i in range(2)]
        correspondences = build_correspondences(*frame_corners, self.point_cloud_builder.ids)

        points = []
        params = TriangulationParameters(REPROJ_ERR, 1.0, 1e-3)
        
        while len(points) < 8:
            points, point_ids, _ = triangulate_correspondences(correspondences, *frame_view_mats,
                                                               self.intrinsic_mat, params)
            
            params = TriangulationParameters(
                    max_reprojection_error=params.max_reprojection_error * 1.02,
                    min_triangulation_angle_deg=params.min_triangulation_angle_deg / 1.02,
                    min_depth=params.min_depth)
        self.point_cloud_builder.add_points(point_ids, points)

    def triangulate(self, frames):
        frame_corners = [self.corner_storage[frames[i]] for i in range(2)]
        frame_view_mats = [self.view_mats[frames[i]] for i in range(2)]
        correspondences = build_correspondences(*frame_corners, self.point_cloud_builder.ids)
        params = TriangulationParameters(REPROJ_ERR, 1.0, 1e-3)
        points, point_ids, median_cos = triangulate_correspondences(correspondences, *frame_view_mats,
                                                               self.intrinsic_mat, params)
        if points.size:
            self.point_cloud_builder.add_points(point_ids, points)

    def track(self, frame):
        frame_corners = self.corner_storage[frame]
        intersect, corners_idx, points_idx = np.intersect1d(frame_corners.ids, self.point_cloud_builder.ids, return_indices=True)

        corners = frame_corners.points[corners_idx]
        corners_ids = frame_corners.ids[corners_idx]
        points = self.point_cloud_builder.points[points_idx]

        if points.shape[0] < 6:
            raise RuntimeError(f"PnP solver: not enough points ({points.shape[0]}) on frame {frame}")

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(points, corners,
                self.intrinsic_mat, None, flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=REPROJ_ERR)

        if not retval:
            raise RuntimeError(f"cv2.solvePnPRansac failed on frame {frame}")
        
        inliers = np.array(inliers).flatten()
        if inliers.size > self.inlier_counts[frame]:
            self.inlier_counts[frame] = inliers.size
        else:
            return
        corners = corners[inliers]
        points = points[inliers]

        retval, rvec, tvec = cv2.solvePnP(points, corners,
                self.intrinsic_mat, None, flags=cv2.SOLVEPNP_ITERATIVE)

        if not retval:
            raise RuntimeError(f"cv2.solvePnP failed on frame {frame}")

        self.view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        cnt = 0
        for prev in range(self.frame_count):
            if not self.solved_frames[prev] or prev == frame:
                continue
            self.triangulate((prev, frame))

    def try_init_known_views(self, frames, common_thr):
        frame_corners = [self.corner_storage[frames[i]] for i in range(2)]
        common = len(np.intersect1d(frame_corners[0].ids, frame_corners[1].ids))
        if common < common_thr:
            return
        correspondences = build_correspondences(*frame_corners)

        essential_mat, mask = cv2.findEssentialMat(
                correspondences.points_1,
                correspondences.points_2,
                self.intrinsic_mat)
        
        if essential_mat is None:
            return

        correspondences = Correspondences(*[
            getattr(correspondences, i)[mask.flatten().astype(bool)]
        for i in ("ids", "points_1", "points_2")])

        homography_mat, mask = cv2.findHomography(
                correspondences.points_1, correspondences.points_2, 
                method=cv2.RANSAC,
                ransacReprojThreshold=1.0,
                confidence=.999)
        if homography_mat is None:
            return

        inliers = np.count_nonzero(mask)
        inliers_frac = inliers / correspondences.ids.size

        if inliers_frac > .6:
            return

        r1, r2, t = cv2.decomposeEssentialMat(essential_mat)
        possible_solutions = [
            np.hstack((r1, t)),
            np.hstack((r1, -t)),
            np.hstack((r2, t)),
            np.hstack((r2, -t))
        ]

        result = None

        for solution in possible_solutions:
            points, point_ids, _ = triangulate_correspondences(
                    correspondences, eye3x4(),
                    solution, self.intrinsic_mat,
                    TriangulationParameters(REPROJ_ERR, 1.0, 1e-4))
            if point_ids.size >= self.best_init_rating:
                self.best_init_rating = point_ids.size
                self.known_views[0] = frames[0], view_mat3x4_to_pose(eye3x4())
                self.known_views[1] = frames[1], view_mat3x4_to_pose(solution)

    def init_known_views(self):
        if self.known_views[0] is not None and self.known_views[1] is not None:
            return

        pairs = []
        for i in range(self.frame_count):
            for j in range(i + 10, min(i + 60, self.frame_count)):
                pairs.append((i, j))

        np.random.seed(2183798173)
        np.random.shuffle(pairs)
        self.best_init_rating = -1

        common_thr = 1000

        while self.best_init_rating == -1:
            for i, pair in enumerate(pairs):
                if self.best_init_rating != -1 and i >= 100:
                    break
                self.try_init_known_views(pair, common_thr)
                print(f"attempt {i}, best = {self.best_init_rating}, frames = {pair}, common_thr = {common_thr}")
            common_thr *= .9

        print(f"initial_frames = {self.initial_frames}")


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    corners_0 = corner_storage[0]
    point_cloud_builder = PointCloudBuilder(corners_0.ids[:1],
                                            np.zeros((1, 3)))
    cam_track = CamTrack(
        corner_storage,
        (known_view_1, known_view_2),
        point_cloud_builder,
        intrinsic_mat
    )

    cam_track.run()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        cam_track.view_mats,
        intrinsic_mat,
        corner_storage,
        REPROJ_ERR
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, cam_track.view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

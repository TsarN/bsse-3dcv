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


@attr.s
class CamTrack:
    corner_storage           = attr.ib()
    known_views              = attr.ib()
    point_cloud_builder      = attr.ib()
    intrinsic_mat            = attr.ib()
    triangulation_parameters = attr.ib()

    @property
    def frame_count(self):
        return len(self.corner_storage)

    @property
    def initial_frames(self):
        return [self.known_views[i][0] for i in range(2)]

    def track_order(self):
        self.init_known_views()

        fst, snd = sorted(self.initial_frames)

        self.start(1)

        for i in range(fst + 1, snd):
            self.steps += 1
            yield i

        for i in range(snd + 1, self.frame_count):
            self.steps += 1
            yield i

        self.start(-1)

        for i in reversed(range(fst)):
            self.steps += 1
            yield i

    def track_prev(self, frame):
        for _ in range(self.steps):
            frame -= self.direction
            yield frame

    def start(self, direction):
        self.direction = direction
        self.steps = 0
        self.view_mats = [pose_to_view_mat3x4(self.known_views[0][1])] * self.frame_count
        self.view_mats[self.initial_frames[1]] = pose_to_view_mat3x4(self.known_views[1][1])
        initial_view_mats = [self.view_mats[self.initial_frames[i]] for i in range(2)]
        camera_centers = [to_camera_center(initial_view_mats[i]) for i in range(2)]
        self.initial_distance = np.linalg.norm(camera_centers[0] - camera_centers[1])
        self.outliers = np.zeros(max([i.ids.size for i in self.corner_storage]) + 1, dtype=bool)

        if direction == 1:
            self.track2(self.initial_frames)

    def track2(self, frames):
        frame_corners = [self.corner_storage[frames[i]] for i in range(2)]
        frame_view_mats = [self.view_mats[frames[i]] for i in range(2)]
        correspondences = build_correspondences(*frame_corners, self.point_cloud_builder.ids)

        points = []
        params = self.triangulation_parameters
        
        while len(points) < 8:
            points, point_ids, _ = triangulate_correspondences(correspondences, *frame_view_mats,
                                                               self.intrinsic_mat, params)
            
            if frames == self.initial_frames:
                params = TriangulationParameters(
                        max_reprojection_error=params.max_reprojection_error * 1.02,
                        min_triangulation_angle_deg=params.min_triangulation_angle_deg / 1.02,
                        min_depth=params.min_depth)
            else:
                break
        self.point_cloud_builder.add_points(point_ids, points)

    def track(self, frame, final=False):
        frame_corners = self.corner_storage[frame]
        intersect, corners_idx, points_idx = np.intersect1d(frame_corners.ids, self.point_cloud_builder.ids, return_indices=True)

        for mask in [self.outliers, np.zeros_like(self.outliers)]:
            ids = np.where(np.invert(mask))
            inliers_idx = np.intersect1d(ids, intersect, return_indices=True)[2]
            corners = frame_corners.points[corners_idx[inliers_idx]]
            corners_ids = frame_corners.ids[corners_idx[inliers_idx]]
            points = self.point_cloud_builder.points[points_idx[inliers_idx]]
            if points.shape[0] >= 6:
                break
        else:
            raise RuntimeError(f"PnP solver: not enough points ({points.shape[0]}) on frame {frame}")

        p = 4.0

        if final:
            retval, rvec, tvec = cv2.solvePnP(points, corners, self.intrinsic_mat, None, flags=cv2.SOLVEPNP_ITERATIVE)

            if not retval:
                raise RuntimeError(f"cv2.solvePnP failed on frame {frame}")
        else:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(points, corners, self.intrinsic_mat, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=p)

            if not retval:
                raise RuntimeError(f"cv2.solvePnPRansac failed on frame {frame}")

        self.view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        if not final:
            while len(inliers) < 5 and p < 100:
                p *= 1.02
                inliers = calc_inlier_indices(points, corners, np.matmul(self.intrinsic_mat, self.view_mats[frame]), p)
            outlier_ids = np.setdiff1d(corners_ids, corners_ids[inliers])
            self.outliers[outlier_ids] = True
            self.track(frame, final=True)

        cnt = 0
        for prev in self.track_prev(frame):
            if not check_baseline(self.view_mats[prev], self.view_mats[frame], self.initial_distance * .15):
                continue
            self.track2((prev, frame))
            cnt += 1
            if cnt >= 5:
                break

    def try_init_known_views(self, frames):
        frame_corners = [self.corner_storage[frames[i]] for i in range(2)]
        correspondences = build_correspondences(*frame_corners)

        essential_mat, mask = cv2.findEssentialMat(correspondences.points_1,
                correspondences.points_2, self.intrinsic_mat)

        correspondences = Correspondences(*[
            getattr(correspondences, i)[mask.flatten().astype(bool)]
        for i in ("ids", "points_1", "points_2")])

        homography_mat, mask = cv2.findHomography(correspondences.points_1, correspondences.points_2, cv2.RANSAC)
        inliers = np.count_nonzero(mask)
        inliers_frac = inliers / correspondences.ids.size

        if inliers_frac > .6:
            return -1, None

        r1, r2, t = cv2.decomposeEssentialMat(essential_mat)
        possible_solutions = [
            np.hstack((r1, t)),
            np.hstack((r1, -t)),
            np.hstack((r2, t)),
            np.hstack((r2, -t))
        ]

        for solution in possible_solutions:
            points, point_ids, _ = triangulate_correspondences(
                    correspondences, eye3x4(),
                    solution, self.intrinsic_mat,
                    self.triangulation_parameters)
            if point_ids.size >= 50:
                return point_ids.size, view_mat3x4_to_pose(solution)

        return -1, None

    def init_known_views(self):
        if self.known_views[0] is not None and self.known_views[1] is not None:
            return

        max_rating = -1
        max_pose = None
        max_idx = None

        pairs = []
        for i in range(self.frame_count):
            for j in range(i + 1, self.frame_count):
                pair = i, j
                common = np.intersect1d(self.corner_storage[i].ids, self.corner_storage[j].ids).size
                if common < 500:
                    break
                rating, pose = self.try_init_known_views(pair)

                if rating > max_rating:
                    max_rating = rating
                    max_pose = pose
                    max_idx = pair
        
        if max_pose is None:
            raise RuntimeError("Failed to initialize known views")
        
        self.known_views = [
            (max_idx[0], view_mat3x4_to_pose(eye3x4())),
            (max_idx[1], max_pose)
        ]

        print(f"Initialized known views; {self.known_views[0][0]} and {self.known_views[1][0]}")
        print(f"Total frames: {self.frame_count}")

    def run(self):
        with click.progressbar(self.track_order(), label="Tracking camera") as bar:
            for frame in bar:
                self.track(frame)


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
        intrinsic_mat,
        TriangulationParameters(7.0, 1.25, 0.0)
    )

    cam_track.run()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        cam_track.view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, cam_track.view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

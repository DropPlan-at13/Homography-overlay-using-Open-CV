from dataclasses import dataclass
import numpy as np


@dataclass
class TrackResult:
    status: str
    polygon: np.ndarray | None
    matches: int
    inliers: int
    inlier_ratio: float
    reproj_error: float


class TrackStateMachine:
    def __init__(self, smooth_alpha=0.35, lost_after_frames=8):
        self.status = "IDLE"
        self.prev_polygon = None
        self.fail_count = 0
        self.smooth_alpha = smooth_alpha
        self.lost_after_frames = lost_after_frames

    def smooth_polygon(self, polygon):
        if polygon is None:
            return self.prev_polygon
        if self.prev_polygon is None:
            self.prev_polygon = polygon.copy()
            return polygon
        smoothed = self.smooth_alpha * polygon + (1.0 - self.smooth_alpha) * self.prev_polygon
        self.prev_polygon = smoothed
        return smoothed

    def update(self, ok, polygon, matches, inliers, inlier_ratio, reproj_error):
        if ok:
            self.status = "TRACKING"
            self.fail_count = 0
            polygon = self.smooth_polygon(polygon)
        else:
            self.fail_count += 1
            if self.fail_count >= self.lost_after_frames:
                self.status = "LOST"
            polygon = self.prev_polygon if self.status == "TRACKING" else None

        return TrackResult(
            status=self.status,
            polygon=polygon,
            matches=matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reproj_error=reproj_error,
        )

    def reset(self):
        self.status = "IDLE"
        self.prev_polygon = None
        self.fail_count = 0

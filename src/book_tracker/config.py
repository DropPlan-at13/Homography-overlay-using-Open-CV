from dataclasses import dataclass


@dataclass
class TrackerConfig:
    detector: str = "SIFT"
    max_features: int = 1200
    ratio_test: float = 0.86
    ransac_thresh: float = 5.0
    min_matches: int = 10
    min_inliers: int = 8
    min_inlier_ratio: float = 0.15
    max_reproj_error: float = 8.0
    bright_threshold: int = 100
    min_mask_pixels: int = 400
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    smooth_alpha: float = 0.35
    lost_after_frames: int = 8
    square_roi_size: int = 550
    temporal_flow_min_points: int = 20
    temporal_flow_win_size: int = 21
    temporal_flow_max_level: int = 3

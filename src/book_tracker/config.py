from dataclasses import dataclass


@dataclass
class TrackerConfig:
    detector: str = "SIFT"
    max_features: int = 800
    ratio_test: float = 0.86
    ransac_thresh: float = 5.0
    min_matches: int = 10
    min_inliers: int = 8
    min_inlier_ratio: float = 0.15
    max_reproj_error: float = 8.0
    bright_threshold: int = 100
    min_mask_pixels: int = 400
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    smooth_alpha: float = 0.35
    lost_after_frames: int = 8
    square_roi_size: int = 550
    temporal_flow_min_points: int = 20
    temporal_flow_win_size: int = 21
    temporal_flow_max_level: int = 3
    keyframe_interval: int = 4
    ecc_interval: int = 8
    ecc_enabled: bool = True
    ecc_min_correlation: float = 0.75
    ecc_max_iterations: int = 60
    ecc_epsilon: float = 1e-5
    ecc_gauss_filter_size: int = 5

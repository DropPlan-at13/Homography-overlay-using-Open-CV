import time

import cv2
import numpy as np

from .capture import open_camera, read_frame
from .config import TrackerConfig
from .features import compute_features as _compute_features, create_detector
from .geometry import homography_from_matches, polygon_is_plausible, reprojection_error
from .matching import create_matcher, match_features as _match_features
from .tracker import TrackResult, TrackStateMachine
from .ui import compose_video_style_view, draw_detection, draw_live_panel, draw_reference_panel_with_roi


WINDOW_NAME = "Book Homography Tracker"


def _empty_result(status="IDLE"):
    return TrackResult(status=status, polygon=None, matches=0, inliers=0, inlier_ratio=0.0, reproj_error=0.0)


def _normalize_roi(x1, y1, x2, y2, width, height):
    left = max(0, min(int(x1), int(x2)))
    right = min(width, max(int(x1), int(x2)))
    top = max(0, min(int(y1), int(y2)))
    bottom = min(height, max(int(y1), int(y2)))
    if right - left < 8 or bottom - top < 8:
        return None
    return left, top, right, bottom


def _roi_to_mask(frame_shape, roi):
    if roi is None:
        return None
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = roi
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


class ROISelector:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image.copy()
        self.dragging = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.roi = None
        self.cancelled = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end = (x, y)
            height, width = self.image.shape[:2]
            self.roi = _normalize_roi(self.start[0], self.start[1], self.end[0], self.end[1], width, height)

    def _render(self):
        view = self.image.copy()
        cv2.putText(
            view,
            "Drag a bounding box around the target object and release the mouse",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            view,
            "Press Q or ESC to cancel selection",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        if self.dragging or self.roi is not None:
            cv2.rectangle(view, self.start, self.end, (0, 255, 0), 2)
        return view

    def run(self):
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        while True:
            cv2.imshow(self.window_name, self._render())
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.cancelled = True
                break
            if self.roi is not None and not self.dragging:
                break
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        if self.cancelled:
            return None
        return self.roi


def select_roi(window_name, image):
    """Freeze the frame and let the user drag a bounding box for the reference object."""
    return ROISelector(window_name, image).run()


def compute_features(detector, image, mask=None):
    return _compute_features(detector, image, mask)


def match_features(matcher, desc_ref, desc_live, ratio=0.75):
    return _match_features(matcher, desc_ref, desc_live, ratio=ratio)


def compute_homography(kp_ref, kp_live, matches, ransac_thresh=5.0):
    if kp_ref is None or kp_live is None or matches is None or len(matches) < 4:
        return None, None
    return homography_from_matches(kp_ref, kp_live, matches, ransac_thresh=ransac_thresh)


def main():
    cfg = TrackerConfig()

    detector, descriptor_kind = create_detector(cfg.detector, cfg.max_features)
    matcher = create_matcher(descriptor_kind)
    state = TrackStateMachine(smooth_alpha=cfg.smooth_alpha, lost_after_frames=cfg.lost_after_frames)

    cap = open_camera(cfg.camera_index, cfg.frame_width, cfg.frame_height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, cfg.display_width, cfg.display_height)

    reference_frame = None
    reference_roi = None
    reference_mask = None
    kp_ref = None
    desc_ref = None

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break

            result = _empty_result(status=state.status)
            kp_live = None
            matches = []
            inlier_mask = None
            live_panel = frame.copy()
            inlier_ratio = 0.0
            reproj = float("inf")
            quad = None

            if reference_frame is not None and kp_ref is not None and desc_ref is not None and reference_roi is not None:
                live_mask = None
                if state.prev_polygon is not None:
                    live_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    tracked_poly = np.asarray(state.prev_polygon, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(live_mask, [tracked_poly], 255)
                    live_mask = cv2.erode(live_mask, np.ones((5, 5), np.uint8), iterations=1)

                kp_live, desc_live = compute_features(detector, frame, live_mask)
                matches = match_features(matcher, desc_ref, desc_live, ratio=cfg.ratio_test)

                if len(matches) > 10:
                    H, inlier_mask = compute_homography(kp_ref, kp_live, matches, ransac_thresh=cfg.ransac_thresh)
                    if H is not None and inlier_mask is not None:
                        inliers = int(inlier_mask.sum())
                        inlier_ratio = float(inliers / len(matches)) if len(matches) > 0 else 0.0
                        x1, y1, x2, y2 = reference_roi
                        ref_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
                        quad = cv2.perspectiveTransform(ref_corners, H)
                        if polygon_is_plausible(quad, frame.shape):
                            reproj = reprojection_error(H, kp_ref, kp_live, matches, inlier_mask)
                            ok = inliers > 10 and reproj <= cfg.max_reproj_error
                            result = state.update(ok, quad, len(matches), inliers, inlier_ratio, reproj)
                        else:
                            result = state.update(False, None, len(matches), inliers, inlier_ratio, float("inf"))
                    else:
                        result = state.update(False, None, len(matches), 0, 0.0, float("inf"))
                else:
                    result = state.update(False, None, len(matches), 0, 0.0, float("inf"))

                live_panel = draw_live_panel(frame, result)
                if result.polygon is not None:
                    live_panel = draw_detection(live_panel, result.polygon, color=(0, 255, 0), alpha=0.18)
            else:
                live_panel = draw_live_panel(frame, result)
                cv2.putText(
                    live_panel,
                    "Press S to freeze a frame and drag an ROI around the object",
                    (18, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            ref_panel = draw_reference_panel_with_roi(reference_frame, (frame.shape[0], frame.shape[1]), reference_roi)

            frame_count += 1
            elapsed = max(1e-6, time.time() - start_time)
            fps = frame_count / elapsed
            cv2.putText(live_panel, f"FPS={fps:.1f}", (18, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(live_panel, f"Detector={cfg.detector}", (18, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(live_panel, "S=select ROI  T=toggle ORB/SIFT  R=reset  Q=quit", (18, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            canvas = compose_video_style_view(
                live_panel,
                ref_panel,
                kp_live=kp_live if matches else None,
                kp_ref=kp_ref,
                matches=matches if matches else [],
                inlier_mask=inlier_mask,
                max_lines=120,
                target_height=frame.shape[0],
                target_half_width=max(320, frame.shape[1] // 2),
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                cfg.detector = "SIFT" if cfg.detector.upper() == "ORB" else "ORB"
                detector, descriptor_kind = create_detector(cfg.detector, cfg.max_features)
                matcher = create_matcher(descriptor_kind)
                reference_frame = None
                reference_roi = None
                reference_mask = None
                kp_ref = None
                desc_ref = None
                state.reset()
                continue
            if key == ord("s"):
                frozen = frame.copy()
                cv2.imshow(WINDOW_NAME, frozen)
                roi = select_roi(WINDOW_NAME, frozen)
                if roi is None:
                    continue

                reference_frame = frozen.copy()
                reference_roi = roi
                reference_mask = _roi_to_mask(reference_frame.shape, roi)
                kp_ref, desc_ref = compute_features(detector, reference_frame, reference_mask)
                if desc_ref is None or kp_ref is None or len(kp_ref) < 4:
                    reference_frame = None
                    reference_roi = None
                    reference_mask = None
                    kp_ref = None
                    desc_ref = None
                    state.reset()
                    continue

                state.reset()
                continue
            if key == ord("r"):
                reference_frame = None
                reference_roi = None
                reference_mask = None
                kp_ref = None
                desc_ref = None
                state.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

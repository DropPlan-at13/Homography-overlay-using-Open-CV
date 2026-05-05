import time
import cv2
import numpy as np

from .capture import open_camera, read_frame
from .config import TrackerConfig
from .features import create_detector, detect_and_describe, create_bright_mask
from .matching import create_matcher, match_ratio_mutual
from .geometry import (
    affine_from_matches,
    affine_from_points,
    homography_from_matches,
    reprojection_error,
    reprojection_error_from_points,
    refine_affine_with_ecc,
    project_reference_corners,
    project_reference_corners_affine,
    polygon_is_plausible,
)
from .tracker import TrackStateMachine, TrackResult
from .ui import draw_reference_panel, draw_live_panel, compose_video_style_view, draw_capture_boundary, create_square_mask, create_polygon_mask


def _empty_result(status="IDLE"):
    return TrackResult(status=status, polygon=None, matches=0, inliers=0, inlier_ratio=0.0, reproj_error=0.0)


def main():
    cfg = TrackerConfig()
    detector, descriptor_kind = create_detector(cfg.detector, cfg.max_features)
    matcher = create_matcher(descriptor_kind)
    state = TrackStateMachine(smooth_alpha=cfg.smooth_alpha, lost_after_frames=cfg.lost_after_frames)

    cap = open_camera(cfg.camera_index, cfg.frame_width, cfg.frame_height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cv2.namedWindow("Book Homography Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Book Homography Tracker", 1600, 900)

    ref_frame = None
    ref_gray = None
    ref_mask = None
    kp_ref = None
    desc_ref = None
    prev_gray = None
    prev_ref_pts = None
    prev_live_pts = None
    frame_index = 0
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break

            result = _empty_result(status=state.status)
            kp_live = None
            matches = []
            inlier_mask = None
            effective_matches = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            use_keyframe_refresh = (
                ref_frame is not None
                and desc_ref is not None
                and (
                    state.status != "TRACKING"
                    or frame_index % cfg.keyframe_interval == 0
                    or prev_live_pts is None
                    or prev_ref_pts is None
                )
            )

            live_mask = None
            if ref_frame is not None and desc_ref is not None:
                if state.status == "TRACKING" and state.prev_polygon is not None:
                    # Once tracking is established, stay inside the object only.
                    live_mask = create_polygon_mask(frame.shape, state.prev_polygon, padding=12)
                else:
                    # During reacquisition, use the bright-object heuristic.
                    live_mask = create_bright_mask(
                        frame,
                        threshold=cfg.bright_threshold,
                        min_mask_pixels=cfg.min_mask_pixels,
                    )

            if ref_frame is not None and desc_ref is not None and use_keyframe_refresh:
                kp_live, desc_live = detect_and_describe(
                    detector,
                    frame,
                    mask=live_mask,
                    bright_threshold=cfg.bright_threshold,
                    min_mask_pixels=cfg.min_mask_pixels,
                )
                matches = match_ratio_mutual(matcher, desc_ref, desc_live, ratio=cfg.ratio_test)
                H, H_mask = homography_from_matches(kp_ref, kp_live, matches, ransac_thresh=cfg.ransac_thresh)
                A, A_mask = affine_from_matches(kp_ref, kp_live, matches, ransac_thresh=cfg.ransac_thresh)

                def _model_score(mask, polygon, reproj_error_value):
                    inliers_local = int(mask.sum()) if mask is not None else 0
                    ratio_local = float(inliers_local / len(matches)) if len(matches) > 0 else 0.0
                    plausible = polygon_is_plausible(polygon, frame.shape)
                    ok_local = (
                        inliers_local >= cfg.min_inliers
                        and ratio_local >= cfg.min_inlier_ratio
                        and reproj_error_value <= cfg.max_reproj_error
                        and plausible
                    )
                    return ok_local, inliers_local, ratio_local

                H_polygon = project_reference_corners(H, ref_frame.shape) if H is not None else None
                H_reproj = reprojection_error(H, kp_ref, kp_live, matches, H_mask)
                H_ok, H_inliers, H_ratio = _model_score(H_mask, H_polygon, H_reproj)

                A_polygon = project_reference_corners_affine(A, ref_frame.shape) if A is not None else None
                A_reproj = reprojection_error(A, kp_ref, kp_live, matches, A_mask)
                A_ok, A_inliers, A_ratio = _model_score(A_mask, A_polygon, A_reproj)

                # Prefer the model that is valid and has better support; affine often wins for in-plane rotation.
                if A_ok and (not H_ok or A_inliers >= H_inliers):
                    polygon = A_polygon
                    reproj = A_reproj
                    inlier_mask = A_mask
                    inliers = A_inliers
                    inlier_ratio = A_ratio
                    model_ok = True
                else:
                    polygon = H_polygon
                    reproj = H_reproj
                    inlier_mask = H_mask
                    inliers = H_inliers
                    inlier_ratio = H_ratio
                    model_ok = H_ok

                # If the chosen model is weak, do not advance tracking.
                if inliers < cfg.min_inliers:
                    model_ok = False
                    inlier_mask = None

                ok = model_ok

                if ok and cfg.ecc_enabled and ref_gray is not None and frame_index % cfg.ecc_interval == 0:
                    ecc_warp, ecc_cc = refine_affine_with_ecc(
                        ref_gray,
                        gray,
                        init_warp=A if A is not None else None,
                        template_mask=ref_mask,
                        input_mask=live_mask,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_max_iterations, cfg.ecc_epsilon),
                        gauss_filt_size=cfg.ecc_gauss_filter_size,
                    )
                    if ecc_warp is not None and ecc_cc >= cfg.ecc_min_correlation:
                        ecc_polygon = project_reference_corners_affine(ecc_warp, ref_frame.shape)
                        if polygon_is_plausible(ecc_polygon, frame.shape):
                            polygon = ecc_polygon
                            state.prev_polygon = ecc_polygon.copy()

                # Temporal refinement/recovery: use KLT optical flow between consecutive frames
                # to keep tracking stable when clockwise rotation weakens descriptor matching.
                if prev_gray is not None and prev_live_pts is not None and prev_ref_pts is not None and len(prev_live_pts) >= cfg.temporal_flow_min_points:
                    next_pts, st, _err = cv2.calcOpticalFlowPyrLK(
                        prev_gray,
                        gray,
                        np.float32(prev_live_pts),
                        None,
                        winSize=(cfg.temporal_flow_win_size, cfg.temporal_flow_win_size),
                        maxLevel=cfg.temporal_flow_max_level,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                    )
                    if next_pts is not None and st is not None:
                        flow_keep = st.ravel().astype(bool)
                        flow_ref_pts = np.float32(prev_ref_pts)[flow_keep]
                        flow_live_pts = np.float32(next_pts)[flow_keep]
                        if len(flow_ref_pts) >= cfg.temporal_flow_min_points:
                            A_flow, A_flow_mask = affine_from_points(flow_ref_pts, flow_live_pts, ransac_thresh=cfg.ransac_thresh)
                            if A_flow is not None and A_flow_mask is not None:
                                flow_inliers = int(A_flow_mask.sum())
                                flow_ratio = float(flow_inliers / len(flow_ref_pts)) if len(flow_ref_pts) > 0 else 0.0
                                flow_polygon = project_reference_corners_affine(A_flow, ref_frame.shape)
                                flow_reproj = reprojection_error_from_points(A_flow, flow_ref_pts, flow_live_pts, A_flow_mask)
                                flow_ok = (
                                    flow_inliers >= cfg.min_inliers
                                    and flow_ratio >= cfg.min_inlier_ratio
                                    and flow_reproj <= cfg.max_reproj_error
                                    and polygon_is_plausible(flow_polygon, frame.shape)
                                )

                                # Prefer the temporal model when descriptor matching is weak or unstable.
                                if flow_ok and (
                                    not ok
                                    or flow_inliers > inliers
                                    or (flow_inliers == inliers and flow_reproj <= reproj)
                                ):
                                    ok = True
                                    polygon = flow_polygon
                                    reproj = flow_reproj
                                    inliers = flow_inliers
                                    inlier_ratio = flow_ratio
                                    inlier_mask = A_flow_mask
                                    matches = []
                                    kp_live = None
                                    effective_matches = len(flow_ref_pts)

                                    # Store the temporal correspondences for the next frame.
                                    prev_ref_pts = flow_ref_pts.reshape(-1, 1, 2)
                                    prev_live_pts = flow_live_pts.reshape(-1, 1, 2)

                        if ok and prev_ref_pts is None and prev_live_pts is None:
                            prev_ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                            prev_live_pts = np.float32([kp_live[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                if ok and inlier_mask is not None and kp_live is not None and len(matches) > 0:
                    inlier_keep = inlier_mask.ravel().astype(bool)
                    inlier_matches = [m for m, keep in zip(matches, inlier_keep) if keep]
                    if len(inlier_matches) >= cfg.min_inliers:
                        prev_ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
                        prev_live_pts = np.float32([kp_live[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 1, 2)
                        effective_matches = len(inlier_matches)

                if matches:
                    effective_matches = len(matches)
                if not matches and prev_ref_pts is not None and prev_live_pts is not None and ok and effective_matches == 0:
                    effective_matches = len(prev_ref_pts)

                result = state.update(ok, polygon, effective_matches, inliers, inlier_ratio, reproj)

                prev_gray = gray
            elif ref_frame is not None and desc_ref is not None and state.status == "TRACKING" and prev_gray is not None and prev_live_pts is not None and prev_ref_pts is not None:
                next_pts, st, _err = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    np.float32(prev_live_pts),
                    None,
                    winSize=(cfg.temporal_flow_win_size, cfg.temporal_flow_win_size),
                    maxLevel=cfg.temporal_flow_max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
                if next_pts is not None and st is not None:
                    flow_keep = st.ravel().astype(bool)
                    flow_ref_pts = np.float32(prev_ref_pts)[flow_keep]
                    flow_live_pts = np.float32(next_pts)[flow_keep]
                    if len(flow_ref_pts) >= cfg.temporal_flow_min_points:
                        A_flow, A_flow_mask = affine_from_points(flow_ref_pts, flow_live_pts, ransac_thresh=cfg.ransac_thresh)
                        if A_flow is not None and A_flow_mask is not None:
                            flow_inliers = int(A_flow_mask.sum())
                            flow_ratio = float(flow_inliers / len(flow_ref_pts)) if len(flow_ref_pts) > 0 else 0.0
                            flow_polygon = project_reference_corners_affine(A_flow, ref_frame.shape)
                            flow_reproj = reprojection_error_from_points(A_flow, flow_ref_pts, flow_live_pts, A_flow_mask)
                            flow_ok = (
                                flow_inliers >= cfg.min_inliers
                                and flow_ratio >= cfg.min_inlier_ratio
                                and flow_reproj <= cfg.max_reproj_error
                                and polygon_is_plausible(flow_polygon, frame.shape)
                            )

                            if flow_ok:
                                result = state.update(True, flow_polygon, len(flow_ref_pts), flow_inliers, flow_ratio, flow_reproj)
                                prev_ref_pts = flow_ref_pts.reshape(-1, 1, 2)
                                prev_live_pts = flow_live_pts.reshape(-1, 1, 2)
                                prev_gray = gray

            panel_ref = draw_reference_panel(ref_frame, (frame.shape[0], frame.shape[1]))
            panel_live = draw_live_panel(frame, result)
            
            # Draw capture boundary if reference not yet captured
            if ref_frame is None:
                panel_live = draw_capture_boundary(panel_live, square_size=cfg.square_roi_size)

            frame_count += 1
            elapsed = max(1e-6, time.time() - t0)
            fps = frame_count / elapsed
            cv2.putText(panel_live, f"FPS={fps:.1f}", (16, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel_live, "Live", (16, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Video-style layout: live on LEFT, reference on RIGHT, with inlier lines.
            # Only draw match lines during active TRACKING state, not during IDLE/LOST
            show_matches = result.status == "TRACKING"
            canvas = compose_video_style_view(
                panel_live,
                panel_ref,
                kp_live=kp_live if show_matches else None,
                kp_ref=kp_ref if show_matches else None,
                matches=matches if show_matches else [],
                inlier_mask=inlier_mask if show_matches else None,
                max_lines=120,
                target_height=frame.shape[0],
                target_half_width=max(320, frame.shape[1] // 2),
            )
            cv2.imshow("Book Homography Tracker", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ref_frame = frame.copy()
                ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                
                # Create square ROI mask (centered, configurable size)
                square_mask = create_square_mask(ref_frame.shape, square_size=cfg.square_roi_size)
                
                # Combine bright mask with square mask (both must be true)
                ref_bright_mask = create_bright_mask(
                    ref_frame,
                    threshold=cfg.bright_threshold,
                    min_mask_pixels=cfg.min_mask_pixels,
                )
                
                # Combine masks: both must be active
                if ref_bright_mask is not None:
                    combined_mask = cv2.bitwise_and(square_mask, ref_bright_mask)
                else:
                    combined_mask = square_mask
                ref_mask = combined_mask
                
                kp_ref, desc_ref = detect_and_describe(
                    detector,
                    ref_frame,
                    mask=combined_mask,
                    bright_threshold=cfg.bright_threshold,
                    min_mask_pixels=cfg.min_mask_pixels,
                )
                state.reset()
                # Start in IDLE; state machine will transition to TRACKING after first successful match
            if key == ord("r"):
                ref_frame = None
                ref_gray = None
                ref_mask = None
                kp_ref = None
                desc_ref = None
                state.reset()

            frame_index += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

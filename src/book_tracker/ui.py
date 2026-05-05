import cv2
import numpy as np


def draw_capture_boundary(frame, square_size=550):
    """Draw a centered square boundary on the frame showing capture region.
    
    Args:
        frame: Input frame
        square_size: Side length of the square in pixels
    
    Returns:
        Frame with dashed square overlay
    """
    view = frame.copy()
    h, w = frame.shape[:2]
    
    # Center the square
    x1 = (w - square_size) // 2
    y1 = (h - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size
    
    # Draw dashed square (using line segments)
    dash_length = 20
    gap = 10
    
    # Top edge
    for x in range(x1, x2, dash_length + gap):
        cv2.line(view, (x, y1), (min(x + dash_length, x2), y1), (0, 255, 255), 2)
    # Bottom edge
    for x in range(x1, x2, dash_length + gap):
        cv2.line(view, (x, y2), (min(x + dash_length, x2), y2), (0, 255, 255), 2)
    # Left edge
    for y in range(y1, y2, dash_length + gap):
        cv2.line(view, (x1, y), (x1, min(y + dash_length, y2)), (0, 255, 255), 2)
    # Right edge
    for y in range(y1, y2, dash_length + gap):
        cv2.line(view, (x2, y), (x2, min(y + dash_length, y2)), (0, 255, 255), 2)
    
    cv2.putText(view, "Press S to capture in square", (x1 + 10, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    return view


def create_square_mask(frame_shape, square_size=550):
    """Create a binary mask with 255 inside the centered square, 0 elsewhere.
    
    Args:
        frame_shape: Shape of the frame (h, w, ...)
        square_size: Side length of the square in pixels
    
    Returns:
        Binary mask
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x1 = (w - square_size) // 2
    y1 = (h - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size
    
    # Clip to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    mask[y1:y2, x1:x2] = 255
    return mask

def create_polygon_mask(frame_shape, polygon, padding=18):
    """Create a filled mask from a projected quadrilateral and expand it slightly.

    The padding helps keep features near the object boundary while still excluding
    most background structure outside the tracked region.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon is None:
        return mask

    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return mask

    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    if padding > 0:
        kernel = np.ones((max(3, padding // 2), max(3, padding // 2)), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def draw_reference_panel(ref_frame, panel_size):
    h, w = panel_size
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if ref_frame is None:
        cv2.putText(panel, "Press S to capture reference", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return panel
    resized = cv2.resize(ref_frame, (w, h))
    cv2.putText(resized, "Reference", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return resized


def draw_live_panel(frame, track_result):
    """Draw live panel with tracking polygon and enhanced metrics."""
    view = frame.copy()
    if track_result.polygon is not None:
        pts = track_result.polygon.astype(np.int32)
        color = (0, 255, 0) if track_result.status == "TRACKING" else (0, 0, 255)
        cv2.polylines(view, [pts], True, color, 3)
    
    status_color = (0, 255, 0) if track_result.status == "TRACKING" else ((0, 255, 255) if track_result.status == "IDLE" else (0, 0, 255))
    
    # Enhanced metrics display
    cv2.putText(view, f"STATUS: {track_result.status}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(view, f"Matches: {track_result.matches} | Inliers: {track_result.inliers}", 
                (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Accuracy metrics with better visibility
    accuracy = (track_result.inlier_ratio * 100) if track_result.inlier_ratio > 0 else 0
    accuracy_color = (0, 255, 0) if accuracy > 50 else ((0, 255, 255) if accuracy > 30 else (0, 0, 255))
    cv2.putText(view, f"Accuracy: {accuracy:.1f}% | Reproj: {track_result.reproj_error:.2f}px", 
                (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, accuracy_color, 2)
    
    return view


def compose_split_screen(ref_panel, live_panel):
    h = max(ref_panel.shape[0], live_panel.shape[0])
    if ref_panel.shape[0] != h:
        ref_panel = cv2.resize(ref_panel, (ref_panel.shape[1], h))
    if live_panel.shape[0] != h:
        live_panel = cv2.resize(live_panel, (live_panel.shape[1], h))
    return np.hstack([ref_panel, live_panel])


def compose_video_style_view(
    live_panel,
    ref_panel,
    kp_live=None,
    kp_ref=None,
    matches=None,
    inlier_mask=None,
    max_lines=120,
    target_height=None,
    target_half_width=None,
):
    """Compose left-live / right-reference canvas and draw cross-panel inlier lines.

    This mimics classic homography demos where green lines connect matched points
    between the live feed and the static reference image.
    """
    # Force exact 50/50 split to avoid uneven panel layout.
    h = target_height if target_height is not None else max(live_panel.shape[0], ref_panel.shape[0])
    w_half = target_half_width if target_half_width is not None else min(live_panel.shape[1], ref_panel.shape[1])

    live_h0, live_w0 = live_panel.shape[:2]
    ref_h0, ref_w0 = ref_panel.shape[:2]

    live_panel = cv2.resize(live_panel, (w_half, h))
    ref_panel = cv2.resize(ref_panel, (w_half, h))

    sx_live = w_half / max(1, live_w0)
    sy_live = h / max(1, live_h0)
    sx_ref = w_half / max(1, ref_w0)
    sy_ref = h / max(1, ref_h0)

    canvas = np.hstack([live_panel, ref_panel])
    x_offset = w_half

    if kp_live is None or kp_ref is None or matches is None or len(matches) == 0:
        return canvas

    keep_mask = None
    if inlier_mask is not None:
        keep_mask = inlier_mask.ravel().astype(bool)

    drawn = 0
    for i, m in enumerate(matches):
        if keep_mask is not None and (i >= len(keep_mask) or not keep_mask[i]):
            continue

        p_live = kp_live[m.trainIdx].pt
        p_ref = kp_ref[m.queryIdx].pt
        p1 = (int(p_live[0] * sx_live), int(p_live[1] * sy_live))
        p2 = (int(p_ref[0] * sx_ref + x_offset), int(p_ref[1] * sy_ref))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, (0, 255, 0), -1, cv2.LINE_AA)

        drawn += 1
        if drawn >= max_lines:
            break

    return canvas

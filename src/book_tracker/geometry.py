import cv2
import numpy as np


def _homography_method():
    # USAC_MAGSAC is more stable than plain RANSAC for outlier-heavy, rotated views.
    return getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def homography_from_matches(kp_ref, kp_live, matches, ransac_thresh=3.0):
    if len(matches) < 4:
        return None, None
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    live_pts = np.float32([kp_live[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    method = _homography_method()
    try:
        H, mask = cv2.findHomography(ref_pts, live_pts, method, ransac_thresh, confidence=0.999, maxIters=5000)
    except TypeError:
        H, mask = cv2.findHomography(ref_pts, live_pts, method, ransac_thresh)
    return H, mask


def affine_from_matches(kp_ref, kp_live, matches, ransac_thresh=3.0):
    """Estimate a rotation-friendly affine transform as a fallback.

    estimateAffinePartial2D models rotation + scale + translation, which is a good
    fallback when the object rotates in-plane and full homography becomes unstable.
    """
    if len(matches) < 3:
        return None, None
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    live_pts = np.float32([kp_live[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    try:
        A, mask = cv2.estimateAffinePartial2D(
            ref_pts,
            live_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=0.999,
            maxIters=5000,
            refineIters=10,
        )
    except TypeError:
        A, mask = cv2.estimateAffinePartial2D(
            ref_pts,
            live_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
        )
    return A, mask


def affine_from_points(ref_pts, live_pts, ransac_thresh=3.0):
    if ref_pts is None or live_pts is None:
        return None, None
    if len(ref_pts) < 3 or len(live_pts) < 3:
        return None, None
    try:
        A, mask = cv2.estimateAffinePartial2D(
            np.float32(ref_pts),
            np.float32(live_pts),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=0.999,
            maxIters=5000,
            refineIters=10,
        )
    except TypeError:
        A, mask = cv2.estimateAffinePartial2D(
            np.float32(ref_pts),
            np.float32(live_pts),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
        )
    return A, mask


def refine_affine_with_ecc(template_gray, input_gray, init_warp=None, template_mask=None, input_mask=None, criteria=None, gauss_filt_size=5):
    """Refine an affine warp with OpenCV ECC inside a masked ROI.

    ECC is an area-based alignment method that can smooth out sparse-match jitter
    and keep the object outline continuous through moderate rotation.
    """
    if template_gray is None or input_gray is None:
        return None, float("-inf")

    template = np.asarray(template_gray)
    image = np.asarray(input_gray)
    if template.ndim != 2 or image.ndim != 2:
        return None, float("-inf")

    if template.dtype != np.float32:
        template = template.astype(np.float32)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if template.max(initial=0.0) > 1.5:
        template /= 255.0
    if image.max(initial=0.0) > 1.5:
        image /= 255.0

    warp = np.eye(2, 3, dtype=np.float32) if init_warp is None else np.asarray(init_warp, dtype=np.float32).copy()
    if warp.shape != (2, 3):
        warp = np.eye(2, 3, dtype=np.float32)

    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)

    try:
        if hasattr(cv2, "findTransformECCWithMask") and template_mask is not None and input_mask is not None:
            cc, warp = cv2.findTransformECCWithMask(
                template,
                image,
                np.asarray(template_mask, dtype=np.uint8),
                np.asarray(input_mask, dtype=np.uint8),
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                gauss_filt_size,
            )
        else:
            cc, warp = cv2.findTransformECC(
                template,
                image,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                input_mask if input_mask is not None else template_mask,
                gauss_filt_size,
            )
    except (cv2.error, TypeError, ValueError):
        return None, float("-inf")

    return warp, float(cc)


def reprojection_error_from_points(T, ref_pts, live_pts, inlier_mask):
    if T is None or inlier_mask is None:
        return float("inf")
    inlier_mask = np.asarray(inlier_mask).ravel().astype(bool)
    if ref_pts is None or live_pts is None or len(ref_pts) == 0 or len(live_pts) == 0:
        return float("inf")
    ref_pts = np.float32(ref_pts)
    live_pts = np.float32(live_pts)
    ref_in = ref_pts[inlier_mask]
    live_in = live_pts[inlier_mask]
    if len(ref_in) < 3:
        return float("inf")
    if T.shape == (2, 3):
        proj = cv2.transform(ref_in.reshape(-1, 1, 2), T)
    else:
        proj = cv2.perspectiveTransform(ref_in.reshape(-1, 1, 2), T)
    err = np.linalg.norm(proj - live_in.reshape(-1, 1, 2), axis=2)
    return float(np.mean(err))


def project_reference_corners_affine(A, ref_shape):
    h, w = ref_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if A is None:
        return None
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    pts = np.hstack([corners, ones])
    warped = pts @ A.T
    return warped.reshape(-1, 1, 2)


def reprojection_error(H, kp_ref, kp_live, matches, inlier_mask):
    if H is None or inlier_mask is None:
        return float("inf")
    inliers = [m for m, keep in zip(matches, inlier_mask.ravel().tolist()) if keep]
    if len(inliers) < 4:
        return float("inf")
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in inliers]).reshape(-1, 1, 2)
    live_pts = np.float32([kp_live[m.trainIdx].pt for m in inliers]).reshape(-1, 1, 2)
    if H.shape == (2, 3):
        proj = cv2.transform(ref_pts, H)
    else:
        proj = cv2.perspectiveTransform(ref_pts, H)
    err = np.linalg.norm(proj - live_pts, axis=2)
    return float(np.mean(err))


def project_reference_corners(H, ref_shape):
    h, w = ref_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    if H is None:
        return None
    return cv2.perspectiveTransform(corners, H)


def polygon_is_plausible(poly, frame_shape):
    if poly is None:
        return False
    pts = poly.reshape(-1, 2)
    area = cv2.contourArea(pts.astype(np.float32))
    if area < 1000:
        return False
    h, w = frame_shape[:2]
    # Allow more slack during rotation; the current gate is too strict for in-plane turns.
    if (pts[:, 0] < -0.45 * w).any() or (pts[:, 0] > 1.45 * w).any():
        return False
    if (pts[:, 1] < -0.45 * h).any() or (pts[:, 1] > 1.45 * h).any():
        return False
    return True

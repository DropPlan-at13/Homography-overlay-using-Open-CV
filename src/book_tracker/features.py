import cv2
import numpy as np


def create_detector(name: str, max_features: int):
    """Create feature detector tuned for distributed keypoint detection across character features.

    SIFT tuning:
    - contrastThreshold=0.01 (vs default 0.04): detects very soft gradients on character faces/bodies
    - edgeThreshold=6 (vs default 10): further penalizes sharp edges, maximizes character curve matches
    - nfeatures increased to 3000: ensures dense coverage despite ultra-soft filtering
    - sigma=1.5 (vs 1.6): slightly tighter scale-space for finer character detail detection

    ORB tuning:
    - fastThreshold=8 (vs default 20): picks up very weak corners on character strokes and gradients
    - edgeThreshold=15 (vs default 31): minimal edge penalty, retains uniform spatial distribution
    """
    name = (name or "SIFT").upper()
    if name == "SIFT":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(
                nfeatures=max(2500, max_features),
                nOctaveLayers=3,
                contrastThreshold=0.005,
                edgeThreshold=5,
                sigma=1.6
            ), "float"
        if hasattr(cv2, "xfeatures2d"):
            return cv2.xfeatures2d.SIFT_create(
                nfeatures=max(2500, max_features),
                nOctaveLayers=3,
                contrastThreshold=0.005,
                edgeThreshold=5,
                sigma=1.6
            ), "float"
        return cv2.ORB_create(nfeatures=max_features), "binary"
    if name == "ORB":
        return cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=8
        ), "binary"
    if name == "AKAZE":
        return cv2.AKAZE_create(), "binary"
    return cv2.ORB_create(nfeatures=max_features), "binary"


def to_gray(image):
    if image is None:
        return None
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_bright_mask(image, threshold=170, min_mask_pixels=800):
    """Create a binary mask for bright target area while avoiding over-masking.

    Strategy:
    - Try a small threshold ladder so dim scenes still capture the book cover.
    - Keep the largest bright connected component.
    - Expand slightly so cover edges/text are not clipped.
    - Return None if mask is too tiny or too broad (fallback to full-frame features).
    """
    gray = to_gray(image)
    if gray is None:
        return None

    h, w = gray.shape[:2]
    img_area = float(h * w)

    for t in (int(threshold), max(0, int(threshold) - 20), max(0, int(threshold) - 35)):
        _, raw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(raw)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

        # Expand ROI a little so the full cover area is retained.
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

        pixels = int(mask.sum() // 255)
        coverage = pixels / img_area
        if pixels < int(min_mask_pixels):
            continue

        # If mask is too broad, likely background/lighting dominated; skip mask.
        if coverage > 0.85:
            return None

        return mask

    return None


def detect_and_describe(detector, image, mask=None, bright_threshold=180, min_mask_pixels=800):
    gray = to_gray(image)
    if gray is None:
        return [], None

    if mask is None:
        mask = create_bright_mask(gray, threshold=bright_threshold, min_mask_pixels=min_mask_pixels)

    kp, desc = detector.detectAndCompute(gray, mask)
    if kp is None:
        kp = []
    return kp, desc

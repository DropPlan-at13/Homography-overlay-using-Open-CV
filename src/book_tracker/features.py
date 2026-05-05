import cv2
import numpy as np


def create_detector(name: str, max_features: int):
    """Create a feature detector for planar object tracking.

    ORB is the default because it is fast on CPU-only laptops. SIFT can be enabled
    when you want more stable matching on textured objects and can afford the extra cost.
    """
    name = (name or "ORB").upper()
    if name == "SIFT":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(
                nfeatures=max(2000, max_features),
                nOctaveLayers=3,
                contrastThreshold=0.01,
                edgeThreshold=10,
                sigma=1.6
            ), "float"
        if hasattr(cv2, "xfeatures2d"):
            return cv2.xfeatures2d.SIFT_create(
                nfeatures=max(2000, max_features),
                nOctaveLayers=3,
                contrastThreshold=0.01,
                edgeThreshold=10,
                sigma=1.6
            ), "float"
        return cv2.ORB_create(nfeatures=max_features), "binary"
    if name == "ORB":
        return cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=10
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


def create_foreground_mask(image, seed_mask, iterations=5):
    """Refine a seed mask into a foreground mask using GrabCut.

    This is run only at reference capture time, so the extra work is acceptable.
    The goal is to keep object features and remove background descriptors before
    matching starts.
    """
    gray = to_gray(image)
    if gray is None or seed_mask is None:
        return seed_mask

    if image.ndim == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color = image.copy()

    gc_mask = np.full(gray.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[seed_mask > 0] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(color, gc_mask, None, bgd_model, fgd_model, int(iterations), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed_mask

    fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    if int(fg_mask.sum() // 255) < 100:
        return seed_mask
    return fg_mask


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


def compute_features(detector, image, mask=None):
    """Compute keypoints and descriptors on an image, optionally constrained by a mask."""
    return detect_and_describe(detector, image, mask=mask)

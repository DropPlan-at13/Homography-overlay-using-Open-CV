import cv2


def create_matcher(descriptor_kind: str):
    if descriptor_kind == "float":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def match_ratio(matcher, desc_ref, desc_live, ratio=0.75):
    if desc_ref is None or desc_live is None:
        return []
    if len(desc_ref) < 2 or len(desc_live) < 2:
        return []
    knn = matcher.knnMatch(desc_ref, desc_live, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def match_ratio_mutual(matcher, desc_ref, desc_live, ratio=0.75):
    """Lowe-ratio match with a mutual-nearest-neighbor consistency check.

    This is more rotation-robust than one-way matching because it removes
    asymmetric correspondences that often appear when the object rotates.
    """
    forward = match_ratio(matcher, desc_ref, desc_live, ratio=ratio)
    if not forward or desc_ref is None or desc_live is None:
        return forward

    if len(desc_ref) < 2 or len(desc_live) < 2:
        return forward

    reverse_knn = matcher.knnMatch(desc_live, desc_ref, k=2)
    reverse_best = {}
    for pair in reverse_knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            reverse_best[m.queryIdx] = m.trainIdx

    mutual = []
    for m in forward:
        # Forward: ref -> live. Reverse map stores live -> ref.
        if reverse_best.get(m.trainIdx) == m.queryIdx:
            mutual.append(m)
    return mutual


def match_features(matcher, desc_ref, desc_live, ratio=0.75):
    """Match descriptors with Lowe's ratio test and mutual consistency."""
    return match_ratio_mutual(matcher, desc_ref, desc_live, ratio=ratio)

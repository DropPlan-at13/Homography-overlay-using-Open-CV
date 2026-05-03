import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from book_tracker.features import create_detector, detect_and_describe
from book_tracker.matching import create_matcher, match_ratio
from book_tracker.geometry import homography_from_matches


def synthetic_book_image():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (520, 340), (255, 255, 255), -1)
    cv2.putText(img, "COMPUTER VISION", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "GEOMETRY", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def test_homography_smoke():
    ref = synthetic_book_image()
    H_gt = np.array([[0.95, 0.10, 30], [-0.05, 1.00, 20], [0.0003, 0.0005, 1.0]], dtype=np.float32)
    live = cv2.warpPerspective(ref, H_gt, (600, 400))

    detector, descriptor_kind = create_detector("SIFT", 1200)
    matcher = create_matcher(descriptor_kind)

    kp1, d1 = detect_and_describe(detector, ref)
    kp2, d2 = detect_and_describe(detector, live)
    matches = match_ratio(matcher, d1, d2, ratio=0.8)
    H, mask = homography_from_matches(kp1, kp2, matches, ransac_thresh=4.0)

    assert H is not None
    assert len(matches) >= 12
    assert mask is not None

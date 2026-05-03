# Book Homography Tracker (Python + OpenCV)

Real-time webcam tracker for a planar book cover using feature matching + RANSAC homography.

## Features
- Press `S` to capture a reference frame (book cover)
- Split-screen UI:
  - Left: live tracking view
  - Right: fixed reference image
- Green cross-panel lines for inlier feature matches (demo-style)
- Robust matching and homography validation
- Status states: `IDLE`, `TRACKING`, `LOST`
- Press `R` to reset reference, `Q` to quit

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Controls
- `S`: capture reference
- `R`: reset reference
- `Q`: quit

## Test (smoke)
```bash
python -m pytest -q
```

## Notes
- Use textured, matte book covers for best results.
- Avoid glossy or plain covers with little text/contrast.

## Tutorial-Style Workflow (Reference Alignment)
This implementation follows the same core flow used in standard OpenCV homography tracking tutorials:

1. Capture reference frame on `S`
2. Detect keypoints/descriptors on reference and live frame
3. Match descriptors with ratio-test filtering
4. Estimate homography with RANSAC
5. Project reference corners to live frame and draw polygon
6. Keep/lose track using inlier and reprojection validation (`TRACKING` / `LOST`)

If you want strict one-to-one alignment with your linked video (same detector, thresholds, and UI style), share the exact settings or key timestamps and I will tune `src/book_tracker/config.py` accordingly.

## Detailed Summary & Analysis

For comprehensive project documentation including:
- Full configuration values and thresholds
- Module descriptions and logic flow
- Troubleshooting guide
- Interview talking points
- Performance metrics and deployment notes

See **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** — suitable for sharing with recruiters, teammates, and code reviewers.

# Book Homography Tracker (Python + OpenCV)

Real-time webcam tracker for a planar book cover or banner using feature matching, robust geometry, and a split-screen OpenCV UI.

## Features
- Press `S` to capture a reference frame
- Square capture region to limit background influence during reference capture
- Split-screen UI:
  - Left: live tracking view
  - Right: fixed reference image
- Green cross-panel lines for inlier feature matches during active tracking
- Homography plus affine fallback for rotation tolerance
- Temporal optical-flow recovery for CPU-only machines
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

## Test
```bash
python -m pytest -q
```

## Notes
- Use textured, matte book covers or banners for best results.
- On CPU-only laptops, the pipeline uses lighter feature settings plus temporal tracking to keep FPS stable.

## Workflow
1. Capture the target inside the square ROI on `S`
2. Detect keypoints/descriptors on reference and live frame
3. Match descriptors with ratio-test and mutual consistency filtering
4. Estimate homography or affine transform with robust estimation
5. Project reference corners to the live frame and draw the polygon
6. Keep or lose track using inlier, reprojection, and temporal validation

## Detailed Summary
For comprehensive project documentation including:
- Configuration values and thresholds
- Module descriptions and logic flow
- Troubleshooting guide
- Interview talking points
- Performance notes

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

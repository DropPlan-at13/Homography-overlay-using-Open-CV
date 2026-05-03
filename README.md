# Book Homography Tracker

Real-time book and banner tracking with Python, OpenCV, feature matching, and robust planar geometry.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Platform](https://img.shields.io/badge/Target-CPU%20Only-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## ✨ Overview

This project tracks a planar object such as a book cover, poster, or banner from a webcam feed.
It captures a reference image, extracts features from the object, and then matches those features
against the live camera stream to estimate pose and draw tracking overlays in real time.

The pipeline is designed to work well on a typical laptop with no GPU. It uses a compact feature
budget, robust matching, affine fallback for rotation, and temporal optical-flow recovery to keep
the tracker responsive under motion.

## 🎯 What It Does

- Captures a reference frame with `S`
- Uses a square ROI to reduce background interference during capture
- Detects and matches keypoints between reference and live frames
- Estimates a planar transform with robust geometry
- Draws live tracking polygons and green cross-panel match lines
- Falls back to temporal tracking when rotation weakens descriptor matching
- Runs comfortably on a CPU-only laptop

## 🧠 Core Techniques

- SIFT-based feature detection for stable keypoints on textured objects
- Mutual nearest-neighbor filtering for more reliable matches
- RANSAC / USAC-style robust transform estimation
- Affine partial transform fallback for in-plane rotation
- KLT optical flow for temporal recovery between frames
- State machine for `IDLE`, `TRACKING`, and `LOST`

## 🖼️ UI Layout

The app uses a split-screen layout:

- Left side: live webcam feed with tracking overlay
- Right side: captured reference image
- Green lines: inlier matches during active tracking
- Yellow square: capture region before reference is stored

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the app

```bash
python main.py
```

### 3. Capture and track

- Place the target inside the square capture region
- Press `S` to capture the reference
- Show the same object again in the live camera
- Observe the polygon, match lines, FPS, and accuracy values

## ⌨️ Controls

- `S` - Capture reference frame
- `R` - Reset the current reference
- `Q` - Quit the application

## 🛠️ Project Structure

```text
main.py
src/book_tracker/
  capture.py      # Camera open/read helpers
  config.py       # Tunable tracker settings
  features.py     # Detector creation and feature extraction
  matching.py     # Descriptor matching logic
  geometry.py     # Homography / affine math and projection
  tracker.py      # Tracking state machine
  ui.py           # Split-screen rendering and overlays
  app.py          # Main runtime loop
tests/
  test_pipeline_smoke.py
```

## ⚙️ Configuration Notes

The tracker is tuned for a CPU-only machine and currently uses:

- a moderate SIFT feature budget for speed
- a square ROI for cleaner reference capture
- mutual matching to reduce false correspondences
- affine fallback when rotation becomes difficult
- temporal optical flow to stabilize the pose across frames

If you want a different tradeoff between speed and accuracy, the main knobs are in `src/book_tracker/config.py`.

## 📈 Tracking Pipeline

1. Read a webcam frame
2. If no reference is stored, show the capture boundary
3. When `S` is pressed, detect features inside the square ROI
4. During live tracking, detect live features and match them to the reference
5. Estimate the transform with robust geometry
6. Validate inliers, reprojection error, and polygon plausibility
7. If descriptor matching weakens, use temporal optical flow to recover
8. Render the split-screen view with overlays and metrics

## 🔍 Accuracy Strategy

This repo uses multiple layers of validation so the tracker behaves better when the object rotates:

- mutual match filtering removes unstable one-way correspondences
- affine fallback handles simple in-plane rotation better than a strict homography in some frames
- temporal tracking reuses previous frame information instead of starting over every time
- inlier and reprojection thresholds reject weak geometry before it becomes visible as drift

## 🧪 Testing

Run the smoke test to confirm the pipeline is working:

```bash
python -m pytest -q
```

The included test generates a synthetic planar object, warps it, and checks that the tracker can still recover a valid transform.

## 💡 Best Results

- Use a textured, matte poster, banner, or book cover
- Keep the capture object inside the square ROI before pressing `S`
- Avoid shiny surfaces with heavy glare
- Try to hold the object flat when capturing the reference
- If the camera is slow, lower the webcam resolution in `src/book_tracker/config.py`

## 🧰 Troubleshooting

- If tracking is unstable, try a better-lit target with more texture.
- If FPS is low, reduce `frame_width`, `frame_height`, or `max_features`.
- If the object is lost during rotation, recapture the reference inside the capture square.
- If you see background matches, tighten the ROI or increase contrast on the target.

## 📚 Documentation

For a deeper breakdown of the configuration, modules, and design decisions, see:

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## 🤝 Sharing the Repo

If you are sharing this project publicly, consider adding:

- a short demo GIF or screenshot
- a sample target image for reference capture
- a small architecture diagram
- a short note about CPU-only performance

That makes the repository easier to understand at a glance and helps reviewers quickly see what the project does.

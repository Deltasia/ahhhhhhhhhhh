# Realtime Webcam Object Tracking

This project streams live frames from a webcam, performs object detection with Ultralytics YOLO, and annotates the feed with distance and angle measurements. A reference pose can be captured to track how far the object moves relative to that point. Camera intrinsics are loaded from `calibration_data.npz` when available, improving the accuracy of the angular and distance estimates.

## Prerequisites

- Windows 10/11
- Python 3.11 (recommended) or 3.10
- A working webcam
- Ultralytics YOLO weights in `models/best.pt` (default)

> **Note**: Avoid Python 3.13 for now—several third-party wheels (Pillow, torch) are not published yet and will fail to build.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirement.txt
```

If you already have a calibrated camera, copy `calibration_data.npz` into the project root. Otherwise, follow the calibration instructions below.

## Running the app

```powershell
.\venv\Scripts\Activate.ps1
python app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser. The dashboard shows:

- **Latest distance/angles** – real-time measurements for the highest-confidence detection.
- **Reference** – the stored baseline measurement.
- **Δ values** – displacement relative to the reference (lateral and forward components).

### Capturing a reference point

1. Position the target so that YOLO detects it (bounding box visible on the stream).
2. Click **Set reference**. The current measurement is saved and rendered as a blue cross.
3. Move the object; the dashboard updates the deltas. Use **Clear reference** to discard the stored pose.

## Calibrating the camera

Accurate distance and angle estimates depend on the camera’s focal length and distortion coefficients. The project looks for `calibration_data.npz` in the workspace root.

### 1. Capture calibration images

Print or display a checkerboard pattern with known square size. The default script expects a 9×6 inner-corner board.

```powershell
python calibrate_camera.py capture --output data/calibration --board-cols 9 --board-rows 6 --square-size-hint 2.4
```

- Press **SPACE** when the checkerboard fills the frame without motion blur.
- Collect at least 8–12 views from different positions and tilts.
- Press **Q** to quit; images are saved to `data\calibration`.

### 2. Run the calibration solver

Measure the square size (in centimeters, millimeters, etc.) and pass it to the solver. The same units will be used for the distance output.

```powershell
python calibrate_camera.py calibrate --images data/calibration --board-cols 9 --board-rows 6 --square-size 2.4 --output calibration_data.npz
```

If successful, the script prints the camera matrix and distortion vector and writes them to `calibration_data.npz`. Restart `app.py` so the new intrinsics are loaded.

### 3. Verifying calibration

- When the app starts, the banner should read **“Camera intrinsics loaded — measurements use calibrated focal length.”**
- Distortions along the edges should be reduced.
- Distance estimates should remain consistent as the object moves horizontally within the frame.

## Changing defaults

Environment variables allow quick overrides:

| Variable | Purpose | Default |
| --- | --- | --- |
| `YOLO_WEIGHTS_PATH` | Path to YOLO weights file | `models/best.pt` |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Capture resolution requested from OpenCV | `640` / `480` |
| `CALIBRATION_PATH` | Location of calibration results | `calibration_data.npz` |
| `KNOWN_WIDTH_CM` | Physical width of the target class | `2.0` |
| `FOCAL_LENGTH_PX` | Override focal length (pixels) without calibration | `1080.0` |
| `PRINCIPAL_POINT_X`, `PRINCIPAL_POINT_Y` | Override image center (pixels) | Frame mid-point |

Adjust `KNOWN_WIDTH_CM` to match the real-world width of the object class you care about (e.g., a marble’s diameter).

### Skipping calibration when intrinsics are known

If you already know your camera’s focal length (in pixels) and principal point, you can skip the checkerboard calibration. Set the overrides before launching `app.py`:

```powershell
$env:FOCAL_LENGTH_PX = 1234.5
$env:PRINCIPAL_POINT_X = 320
$env:PRINCIPAL_POINT_Y = 240
python app.py
```

Leave out `PRINCIPAL_POINT_X/Y` if the optical center is close to the frame midpoint. When `FOCAL_LENGTH_PX` is provided, the app ignores values from `calibration_data.npz` and uses your manual number.

## Troubleshooting

- **No detection / blank stream**: Ensure the webcam is not in use by another application and that YOLO weights exist.
- **Measurements stuck at fallback**: Double-check that `calibration_data.npz` is in the project root and readable.
- **Large distance errors**: Confirm that the physical width (`KNOWN_WIDTH_CM`) matches the object, and re-run calibration with more diverse checkerboard poses.

Happy building! Capture a reference, watch the delta values update in real time, and iterate on your tracking logic as needed.

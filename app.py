import math
import os
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
from ultralytics import YOLO

# ----กันกล้องเอ๋อ-----
import platform

def get_camera_safe(preferred_index=1, fallback_index=0, width=640, height=480):
    # Define backends based on operating system
    system = platform.system()
    if system == "Darwin":  # macOS
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
    elif system == "Linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    
    for idx in [preferred_index, fallback_index]:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # Test camera by reading frames
                for _ in range(10):  # Reduced from 30 for faster startup
                    cap.read()

                ret, _ = cap.read()
                if ret:
                    print(f"Using camera index {idx} with backend {backend} on {system}")
                    return cap
                else:
                    cap.release()
            except Exception as e:
                print(f"Failed to open camera {idx} with backend {backend}: {e}")
                if cap:
                    cap.release()
                continue
    
    raise RuntimeError("ไม่สามารถเปิดกล้องได้ทั้ง preferred และ fallback index")

cap = get_camera_safe(preferred_index=0, fallback_index=0, width=640, height=480)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -----เลือกโมเดล-----
MODEL_WEIGHTS = os.getenv("YOLO_WEIGHTS_PATH", "models/best.pt")
model = YOLO(MODEL_WEIGHTS)

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "calibration_data.npz")
KNOWN_WIDTH_CM = float(os.getenv("KNOWN_WIDTH_CM", "3.0"))
PORT = int(os.getenv("PORT", "8080"))

# ----เอาจากคาลิเบด------
def load_calibration(path: str):
    if not os.path.exists(path):
        print(f"Calibration file '{path}' not found. Using fallback camera parameters.")
        return None, None

    try:
        with np.load(path) as data:
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
            print(f"Loaded calibration data from {path}")
            return camera_matrix, dist_coeffs
    except Exception as exc:
        print(f"Failed to load calibration data: {exc}. Using fallback parameters.")
        return None, None

camera_matrix, dist_coeffs = load_calibration(CALIBRATION_PATH)
manual_focal = os.getenv("FOCAL_LENGTH_PX")
manual_pp_x = os.getenv("PRINCIPAL_POINT_X")
manual_pp_y = os.getenv("PRINCIPAL_POINT_Y")

principal_point = (FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0)
if manual_pp_x is not None and manual_pp_y is not None:
    try:
        principal_point = (float(manual_pp_x), float(manual_pp_y))
        print(
            "Using principal point override from environment variables: "
            f"({principal_point[0]:.2f}, {principal_point[1]:.2f})"
        )
    except ValueError:
        print("Invalid PRINCIPAL_POINT_X/Y override. Falling back to frame center.")

focal_length_px = 1080.0
if manual_focal is not None:
    try:
        focal_length_px = float(manual_focal)
        print(f"Using focal length override from environment: {focal_length_px:.2f} px")
    except ValueError:
        print("Invalid FOCAL_LENGTH_PX override. Falling back to default 1080 px.")

if camera_matrix is not None and manual_focal is None:
    focal_length_px = float(camera_matrix[0, 0])
if camera_matrix is not None and not (manual_pp_x and manual_pp_y):
    principal_point = (float(camera_matrix[0, 2]), float(camera_matrix[1, 2]))

measurement_lock = threading.Lock()
latest_measurement: Optional[Dict[str, Any]] = None
reference_measurement: Optional[Dict[str, Any]] = None


def measurement_to_cartesian(measurement):
    angle_rad = math.radians(measurement["angle_deg"])
    distance = measurement["distance_cm"]
    x = distance * math.sin(angle_rad)
    z = distance * math.cos(angle_rad)
    return x, z


def compute_offsets(current, reference):
    if not current or not reference:
        return None

    curr_x, curr_z = measurement_to_cartesian(current)
    ref_x, ref_z = measurement_to_cartesian(reference)

    return {
        "delta_distance_cm": current["distance_cm"] - reference["distance_cm"],
        "delta_angle_deg": current["angle_deg"] - reference["angle_deg"],
        "delta_x_cm": curr_x - ref_x,
        "delta_z_cm": curr_z - ref_z,
    }


def gen_frames():
    global latest_measurement

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if camera_matrix is not None and dist_coeffs is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        frame_h = frame.shape[0]
        # ----ลองแก้เออเร่อ-----
        results = model(frame,imgsz=640)[0]

        best_detection = None

        if hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])
                # ----น่าจะดีขึ้น-----
                if confidence <= 0.7:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                # -----เดี๋ยวน้องเหงา-------
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{model.names[cls]} {confidence:.2f}",
                            (int(x1), max(int(y1) - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # ------------------------

                if best_detection is None or confidence > best_detection["confidence"]:
                    best_detection = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence,
                        "cls": cls,
                    }

        offsets = None
        active_reference = None

        if best_detection:
            x1 = best_detection["x1"]
            y1 = best_detection["y1"]
            x2 = best_detection["x2"]
            y2 = best_detection["y2"]
            confidence = best_detection["confidence"]
            cls = best_detection["cls"]

            obj_center_x = (x1 + x2) / 2.0
            obj_center_y = (y1 + y2) / 2.0
            obj_width_px = max((x2 - x1), 1e-6)

            distance_cm = (KNOWN_WIDTH_CM * focal_length_px) / obj_width_px

            cx, cy = principal_point
            angle_rad = math.atan2(obj_center_x - cx, focal_length_px)
            angle_deg = math.degrees(angle_rad)
            vertical_angle_deg = math.degrees(math.atan2(obj_center_y - cy, focal_length_px))

            label = f"{model.names[cls]} {confidence:.2f} D:{distance_cm:.1f}cm θ:{angle_deg:.1f}°"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), max(int(y1) - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            measurement = {
                "timestamp": time.time(),
                "label": model.names[cls],
                "confidence": confidence,
                "distance_cm": float(distance_cm),
                "angle_deg": float(angle_deg),
                "vertical_angle_deg": float(vertical_angle_deg),
                "center": {"x": float(obj_center_x), "y": float(obj_center_y)},
                "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                "cartisian": {"x_cm": None, "z_cm": None},
            }

            with measurement_lock:
                latest_measurement = measurement
                offsets = compute_offsets(latest_measurement, reference_measurement)
                active_reference = (
                    reference_measurement.copy() if reference_measurement is not None else None
                )

            if active_reference:
                ref_center = active_reference["center"]
                cv2.drawMarker(frame, (int(ref_center["x"]), int(ref_center["y"])), (255, 0, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.putText(frame, "REF", (int(ref_center["x"]) - 20, int(ref_center["y"]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if offsets:
                offset_text = (f"ΔD:{offsets['delta_distance_cm']:.1f}cm "
                               f"Δθ:{offsets['delta_angle_deg']:.1f}°")
                cv2.putText(frame, offset_text, (10, frame_h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        status_text = "Calibrated" if camera_matrix is not None else "Using fallback intrinsics"
        cv2.putText(frame, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def api_state():
    with measurement_lock:
        latest = latest_measurement.copy() if latest_measurement else None
        reference = reference_measurement.copy() if reference_measurement else None
        offsets = compute_offsets(latest, reference) if latest and reference else None

    return jsonify({
        "latest": latest,
        "reference": reference,
        "offsets": offsets,
        "calibrated": camera_matrix is not None,
        "focal_length_px": focal_length_px,
        "principal_point": {"x": principal_point[0], "y": principal_point[1]},
        "cartesian": {"x_cm": latest["cartisian"]["x_cm"] if latest else None,
                      "z_cm": latest["cartisian"]["z_cm"] if latest else None},
    })


@app.route('/api/reference', methods=['POST', 'DELETE'])
def api_reference():
    global reference_measurement

    with measurement_lock:
        if request.method == 'POST':
            if not latest_measurement:
                return jsonify({
                    "status": "error",
                    "message": "No detection available to store as reference."
                }), 400

            reference_measurement = latest_measurement.copy()
            return jsonify({
                "status": "ok",
                "reference": reference_measurement,
            })

        reference_measurement = None
        return jsonify({"status": "ok", "message": "Reference cleared."})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False)

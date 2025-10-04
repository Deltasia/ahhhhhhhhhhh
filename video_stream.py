import cv2
from camera import Camera
from model import Model
from measurement import MeasurementService

class VideoStreamer:
    def __init__(self, camera: Camera, model: Model, measurement_service: MeasurementService):
        self.camera = camera
        self.model = model
        self.measurement_service = measurement_service

    def gen_frames(self):
        while True:
            ret, frame = self.camera.cap.read()
            if not ret:
                continue

            if self.camera.camera_matrix is not None and self.camera.dist_coeffs is not None:
                frame = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coeffs)

            frame_h = frame.shape[0]
            results = self.model.model(frame, imgsz=640)[0]

            best_detection = None

            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    confidence = float(box.conf[0])
                    if confidence <= 0.7:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"{self.model.model.names[cls]} {confidence:.2f}",
                                (int(x1), max(int(y1) - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                self.measurement_service.update_measurement(best_detection)

                with self.measurement_service.measurement_lock:
                    offsets = self.measurement_service.compute_offsets(
                        self.measurement_service.latest_measurement, self.measurement_service.reference_measurement
                    ) if self.measurement_service.latest_measurement and self.measurement_service.reference_measurement else None
                    active_reference = (
                        self.measurement_service.reference_measurement.copy() if self.measurement_service.reference_measurement is not None else None
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

            status_text = "Calibrated" if self.camera.camera_matrix is not None else "Using fallback intrinsics"
            cv2.putText(frame, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
import cv2
import time
from collections import Counter
from camera import Camera
from model import Model
from measurement import MeasurementService

class VideoStreamer:
    def __init__(self, camera: Camera, model: Model, measurement_service: MeasurementService, socketio, config):
        self.camera = camera
        self.model = model
        self.measurement_service = measurement_service
        self.socketio = socketio
        self.config = config
        self.last_inference_time = 0

    def gen_frames(self):
        try:
            while True:
                start_time = time.time()
                if not self.camera.cap or not self.camera.cap.isOpened():
                    break
                
                ret, frame = self.camera.cap.read()
                if not ret:
                    continue

                if self.camera.camera_matrix is not None and self.camera.dist_coeffs is not None:
                    frame = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coeffs)

                frame_h = frame.shape[0]

                current_time = time.time()
                do_inference = current_time - self.last_inference_time >= 1 / self.config.MAX_INFERENCE_FPS

                best_detection = None

                offsets = None
                active_reference = None

                if do_inference:
                    self.last_inference_time = current_time

                    results = self.model.model(frame, imgsz=640, verbose=False)[0]

                    # Construct log string manually
                    h, w = frame.shape[:2]
                    if results.boxes is None or len(results.boxes) == 0:
                        det_str = "(no detections)"
                    else:
                        cls_counts = Counter(int(box.cls[0]) for box in results.boxes)
                        det_parts = []
                        for cls, count in cls_counts.items():
                            name = self.model.model.names[cls]
                            det_parts.append(f"{count} {name}")
                        det_str = " ".join(det_parts)

                    speed = results.speed
                    log = (f"0: {w}x{h} {det_str}, {speed['inference']:.1f}ms\n"
                           f"Speed: {speed['preprocess']:.1f}ms preprocess, {speed['inference']:.1f}ms inference, {speed['postprocess']:.1f}ms postprocess per image at shape (1, 3, {h}, {w})")
                    self.socketio.emit('log', log)

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
                elapsed = time.time() - start_time
                sleep_time = max(0, 1 / self.config.MAX_FPS - elapsed)
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Video streaming error: {e}")
        finally:
            # Ensure proper cleanup
            if hasattr(self, 'camera') and self.camera and self.camera.cap:
                self.camera.cap.release()
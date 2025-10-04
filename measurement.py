import math
import threading
import time
from typing import Any, Dict, Optional

from camera import Camera
from model import Model

class MeasurementService:
    def __init__(self, camera: Camera, model: Model):
        self.camera = camera
        self.model = model
        self.measurement_lock = threading.Lock()
        self.latest_measurement: Optional[Dict[str, Any]] = None
        self.reference_measurement: Optional[Dict[str, Any]] = None

    def measurement_to_cartesian(self, measurement):
        angle_rad = math.radians(measurement["angle_deg"])
        distance = measurement["distance_cm"]
        x = distance * math.sin(angle_rad)
        z = distance * math.cos(angle_rad)
        return x, z

    def compute_offsets(self, current, reference):
        if not current or not reference:
            return None

        curr_x, curr_z = self.measurement_to_cartesian(current)
        ref_x, ref_z = self.measurement_to_cartesian(reference)

        return {
            "delta_distance_cm": current["distance_cm"] - reference["distance_cm"],
            "delta_angle_deg": current["angle_deg"] - reference["angle_deg"],
            "delta_x_cm": curr_x - ref_x,
            "delta_z_cm": curr_z - ref_z,
        }

    def get_state(self):
        with self.measurement_lock:
            latest = self.latest_measurement.copy() if self.latest_measurement else None
            reference = self.reference_measurement.copy() if self.reference_measurement else None
            offsets = self.compute_offsets(latest, reference) if latest and reference else None

        return {
            "latest": latest,
            "reference": reference,
            "offsets": offsets,
        }

    def set_reference(self):
        with self.measurement_lock:
            if not self.latest_measurement:
                return False, "No detection available to store as reference."

            self.reference_measurement = self.latest_measurement.copy()
            return True, self.reference_measurement

    def clear_reference(self):
        with self.measurement_lock:
            self.reference_measurement = None
            return True, "Reference cleared."

    def update_measurement(self, best_detection):
        if not best_detection:
            return

        x1 = best_detection["x1"]
        y1 = best_detection["y1"]
        x2 = best_detection["x2"]
        y2 = best_detection["y2"]
        confidence = best_detection["confidence"]
        cls = best_detection["cls"]

        obj_center_x = (x1 + x2) / 2.0
        obj_center_y = (y1 + y2) / 2.0
        obj_width_px = max((x2 - x1), 1e-6)
        
        # distance_cm = ((self.camera.config.KNOWN_WIDTH_CM * self.camera.focal_length_px) / obj_width_px) 
        distance_cm = (0.714*(self.camera.config.KNOWN_WIDTH_CM * self.camera.focal_length_px) / obj_width_px)-3.571

        cx, cy = self.camera.principal_point
        angle_rad = math.atan2(obj_center_x - cx, self.camera.focal_length_px)
        angle_deg = math.degrees(angle_rad)
        vertical_angle_deg = math.degrees(math.atan2(obj_center_y - cy, self.camera.focal_length_px))

        measurement = {
            "timestamp": time.time(),
            "label": self.model.model.names[cls],
            "confidence": confidence,
            "distance_cm": float(distance_cm),
            "angle_deg": float(angle_deg),
            "vertical_angle_deg": float(vertical_angle_deg),
            "center": {"x": float(obj_center_x), "y": float(obj_center_y)},
            "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            "cartisian": {"x_cm": None, "z_cm": None},
        }

        with self.measurement_lock:
            self.latest_measurement = measurement
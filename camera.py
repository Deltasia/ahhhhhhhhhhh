import os
import cv2
import numpy as np
import platform
from config import Config

class Camera:
    def __init__(self, config: Config):
        self.config = config
        self.cap = self.get_camera_safe()
        self.camera_matrix, self.dist_coeffs = self.load_calibration()
        self.principal_point = self.calculate_principal_point()
        self.focal_length_px = self.calculate_focal_length()

    def get_camera_safe(self, preferred_index=0, fallback_index=0, width=640, height=480):
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

        cap = None
        for idx in [preferred_index, fallback_index]:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # Test camera by reading frames
                    for _ in range(10):
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

    def load_calibration(self):
        path = self.config.CALIBRATION_PATH
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

    def calculate_principal_point(self):
        pp = (self.config.FRAME_WIDTH / 2.0, self.config.FRAME_HEIGHT / 2.0)
        if self.config.manual_pp_x is not None and self.config.manual_pp_y is not None:
            try:
                pp = (float(self.config.manual_pp_x), float(self.config.manual_pp_y))
                print(f"Using principal point override: ({pp[0]:.2f}, {pp[1]:.2f})")
            except ValueError:
                print("Invalid PRINCIPAL_POINT_X/Y override. Falling back to frame center.")
        return pp

    def calculate_focal_length(self):
        focal = 1080.0
        if self.config.manual_focal is not None:
            try:
                focal = float(self.config.manual_focal)
                print(f"Using focal length override: {focal:.2f} px")
            except ValueError:
                print("Invalid FOCAL_LENGTH_PX override. Falling back to default 1080 px.")

        if self.camera_matrix is not None and self.config.manual_focal is None:
            focal = float(self.camera_matrix[0, 0])
        if self.camera_matrix is not None and not (self.config.manual_pp_x and self.config.manual_pp_y):
            self.principal_point = (float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2]))
        return focal

    def release(self):
        if self.cap:
            self.cap.release()
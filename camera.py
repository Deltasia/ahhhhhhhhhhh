import os
import cv2
import numpy as np
import platform
import time
from config import Config

# Global cache for camera enumeration
_camera_cache = None
_cache_timestamp = 0
_cache_duration = 30  # Cache for 30 seconds

def enumerate_cameras(max_cameras=10, use_cache=True):
    """Enumerate available cameras and return their indices and names."""
    global _camera_cache, _cache_timestamp
    
    # Check if we should use cached result
    if use_cache and _camera_cache is not None:
        if time.time() - _cache_timestamp < _cache_duration:
            return _camera_cache
    
    available_cameras = []
    system = platform.system()
    
    # Define backends based on operating system
    if system == "Darwin":  # macOS
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
    elif system == "Linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    for idx in range(max_cameras):
        camera_found = False
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # Test if we can actually read from the camera
                    ret, _ = cap.read()
                    if ret:
                        # Try to get camera name/description
                        camera_name = f"Camera {idx}"
                        try:
                            # Some backends support getting device name
                            if hasattr(cap, 'getBackendName'):
                                backend_name = cap.getBackendName()
                                camera_name = f"Camera {idx} ({backend_name})"
                        except Exception:
                            pass
                        
                        available_cameras.append({
                            'index': idx,
                            'name': camera_name,
                            'backend': backend
                        })
                        cap.release()
                        camera_found = True
                        consecutive_failures = 0
                        break  # Found working camera with this index, move to next
                cap.release()
            except Exception:
                # Continue trying other backends
                continue
        
        # If no camera found at this index, increment failure counter
        if not camera_found:
            consecutive_failures += 1
            # Stop searching if we've had too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                break
    
    # Cache the result
    if use_cache:
        _camera_cache = available_cameras
        _cache_timestamp = time.time()
    
    return available_cameras

def clear_camera_cache():
    """Clear the camera enumeration cache."""
    global _camera_cache, _cache_timestamp
    _camera_cache = None
    _cache_timestamp = 0

class Camera:
    def __init__(self, config: Config, camera_index=0):
        self.config = config
        self.camera_index = camera_index
        self.cap = None
        self.cap = self.get_camera_safe(preferred_index=camera_index)
        self.camera_matrix, self.dist_coeffs = self.load_calibration()
        self.principal_point = self.calculate_principal_point()
        self.focal_length_px = self.calculate_focal_length()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.release()

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
    
    def switch_camera(self, new_index):
        """Switch to a different camera index."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_index = new_index
        self.cap = self.get_camera_safe(preferred_index=new_index)
        
        # Clear camera cache after successful switch
        if self.cap is not None:
            clear_camera_cache()
        
        return self.cap is not None

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
            self.cap = None
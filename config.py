import os

class Config:
    def __init__(self):
        self.MODEL_WEIGHTS = os.getenv("YOLO_WEIGHTS_PATH", "models/best.pt")
        self.FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
        self.FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
        self.CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "calibration_data.npz")
        self.KNOWN_WIDTH_CM = float(os.getenv("KNOWN_WIDTH_CM", "3.0"))
        self.PORT = int(os.getenv("PORT", "8080"))
        self.manual_focal = os.getenv("FOCAL_LENGTH_PX")
        self.manual_pp_x = os.getenv("PRINCIPAL_POINT_X")
        self.manual_pp_y = os.getenv("PRINCIPAL_POINT_Y")
        self.MAX_INFERENCE_FPS = float(os.getenv("MAX_INFERENCE_FPS", "10.0"))
        self.MAX_FPS = float(os.getenv("MAX_FPS", "30.0"))
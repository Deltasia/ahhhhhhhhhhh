from ultralytics import YOLO
from config import Config

class Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(self.config.MODEL_WEIGHTS)
from picamera2 import Picamera2, Preview
import cv2
from ultralytics import YOLO

class Camera:
    def __init__(self):
        self.cam = Picamera2()
        self.preview_config = self.cam.create_preview_configuration(main={'size': (384, 512), "format": 'RGB888'})
        self.cam.configure(self.preview_config)
        self.cam.start()
        self.ncnn = YOLO('./yolo11n_ncnn_model')
    def extract(self):
        frame = self.cam.capture_array()
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    def process(self, frame):
        results = self.ncnn(frame)
        annotated_frame = results[0].plot() 
        return annotated_frame
    def display(self, frame):
        cv2.imshow('Prediction Feed', frame)
    def close(self):
        self.cam.stop()
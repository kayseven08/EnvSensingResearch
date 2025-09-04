from picamera2 import Picamera2, Preview
import cv2
from ultralytics import YOLO

cam = Picamera2()

preview_config = cam.create_preview_configuration(main={'size': (640,480), "format": 'RGB888'})
cam.configure(preview_config)
89
# cam.start_preview(Preview.QTGL)

cam.start()

model = YOLO('yolov8n.pt')

while True:
    frame = cam.capture_array()
    processed_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("Camera Feed", processed_frame) 
    results = model(processed_frame)
    annotated_frame = results[0].plot()
    cv2.imshow('Prediction Feed', annotated_frame)   
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.stop()
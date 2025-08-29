from picamera2 import Picamera2, Preview
import cv2
import time

cam = Picamera2()

preview_config = cam.create_preview_configuration(main={'size': (640,480)})
cam.configure(preview_config)

cam.start_preview(Preview.QTGL)

cam.start()

while True:
    frame = cam.capture_array()
    processed_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cv2.imshow("Camera Feed", frame) 
    cv2.imshow('Processed Feed', processed_frame)   


cam.stop()
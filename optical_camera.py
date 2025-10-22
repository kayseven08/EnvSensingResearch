from picamera2 import Picamera2, Preview
import cv2
from ultralytics import YOLO

cam = Picamera2()

preview_config = cam.create_preview_configuration(main={'size': (640,480), "format": 'RGB888'})
cam.configure(preview_config)

# cam.start_preview(Preview.QTGL)

cam.start()

# model = YOLO('yolo11n.pt')
# model.export(format="ncnn")
ncnn_model = YOLO('./yolo11n_ncnn_model')
pose_model = YOLO('./yolo11n-pose_ncnn_model')


while True:
    frame = cam.capture_array()
    processed_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imshow("Camera Feed", processed_frame) 
    # results = ncnn_model(processed_frame)
    pose_results = pose_model(processed_frame)
    # annotated_frame = results[0].plot()
    # obscured_frame = results[0].plot(img=bg_image_array)
    pose_frame = pose_results[0].plot()
    # cv2.imshow('Prediction Feed', annotated_frame)
    cv2.imshow('Pose Feed', pose_frame)
    # cv2.imshow('Obscured Feed', obscured_frame)  
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.stop()
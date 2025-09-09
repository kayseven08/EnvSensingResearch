from picamera2 import Picamera2, Preview
import cv2
from ultralytics import YOLO
import supervision as sv
from PIL import Image
from numpy import asarray

cam = Picamera2()

preview_config = cam.create_preview_configuration(main={'size': (540,960), "format": 'RGB888'})
cam.configure(preview_config)

# cam.start_preview(Preview.QTGL)

cam.start()

# model = YOLO('yolo11n.pt')
# model.export(format="ncnn")
ncnn_model = YOLO('./yolo11n-pose_ncnn_model')
# box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

bg_img = Image.open('black_image.jpeg')
bg_image_array = asarray(bg_img)

while True:
    frame = cam.capture_array()
    processed_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow("Camera Feed", processed_frame) 
    results = ncnn_model(processed_frame)
    annotated_frame = results[0].plot()
    obscured_frame = results[0].plot(img=bg_image_array)
    cv2.imshow('Prediction Feed', annotated_frame)
    cv2.imshow('Obscured Feed', obscured_frame)
    # for result in results:
    #     xy = result.keypoints.xy
    #     xyn = result.keypoints.xyn
    #     kpts = result.keypoints.data
    #     print(xy)
    #     print(xyn)
    #     print(kpts)   
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.stop()
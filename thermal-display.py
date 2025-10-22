import depthai as dai
import cv2

THERMAL_IMAGE_BRIGHTNESS_STEP = 10

with dai.Pipeline(True) as pipeline:
    thermal = pipeline.create(dai.node.Thermal)
    thermalImgOut = thermal.color.createOutputQueue()
    thermalConfIn = thermal.inputConfig.createInputQueue()
    thermalConf = dai.ThermalConfig()

    pipeline.start()
    cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Color Map", cv2.WINDOW_NORMAL)
    initialRescaleDone = False
    while True:
        thermalImg = thermalImgOut.get()
        if not initialRescaleDone:
            cv2.resizeWindow("Grayscale", 640, 640)
            cv2.resizeWindow("Color Map", 640, 640)
            initialRescaleDone = True
        thermal_frame = thermalImg.getCvFrame()
        colored_frame = cv2.applyColorMap(thermal_frame, cv2.COLORMAP_PLASMA)
        cv2.imshow("Grayscale", thermal_frame)
        cv2.imshow("Color Map", colored_frame)
        key = cv2.waitKey(1)
        changed = False
        if key == ord("q"):
            break

import depthai as dai, cv2

class Thermal:
    def __init__(self):
        self.pipeline = dai.Pipeline(True)
        self.device = dai.Device(self.pipeline)
        self.thermal = self.pipeline.create(dai.node.Thermal)
        self.thermalImgOut = self.thermal.color.createOutputQueue()
        self.pipeline.start()
        cv2.namedWindow("Color Map", cv2.WINDOW_NORMAL)
    def extract(self):
        thermalImg = self.thermalImgOut.get()
        return thermalImg.getCvFrame()
    def process(self, frame):
        return cv2.applyColorMap(frame, cv2.COLORMAP_PLASMA)
    def display(self, frame):
        cv2.imshow('Color Map', frame)
    def close(self):
        self.device.close()
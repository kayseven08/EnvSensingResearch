import depthai as dai
import cv2

class Thermal:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.device = None
        self.thermal = self.pipeline.create(dai.node.Thermal)
        cv2.namedWindow("Color Map", cv2.WINDOW_NORMAL)

    def start(self):
        self.thermalImgOut = self.thermal.color.createOutputQueue()
        self.pipeline.start()
    def extract(self):
        thermalImgPacket = self.thermalImgOut.tryGet() 
        if thermalImgPacket is not None:
            return thermalImgPacket.getCvFrame() 
        else:
            return None

    def process(self, frame):
        if frame is None:
            return None
        return cv2.applyColorMap(frame, cv2.COLORMAP_PLASMA)

    def display(self, frame):
        if frame is not None:
            cv2.imshow('Color Map', frame)

    def run(self):
        while True:
            frame = self.extract()
            processed_frame = self.process(frame)
            self.display(processed_frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

    def close(self):
        if self.device is not None:
            self.device.close()
        cv2.destroyAllWindows()
        print("Device closed.")

if __name__ == "__main__":
    detector = Thermal()
    try:
        detector.start()
        detector.run()
    except RuntimeError as e:
        print(f"Failed to run: {e}")
    finally:
        detector.close()
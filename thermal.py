import depthai as dai
import cv2

class Thermal:
    #Initialize the DepthAI pipeline
    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.device = None
        self.thermal = self.pipeline.create(dai.node.Thermal)
        # cv2.namedWindow("Color Map", cv2.WINDOW_NORMAL)
        self.start()
    #starts the pipeline and initializes the output queue
    def start(self):
        self.thermalImgOut = self.thermal.color.createOutputQueue()
        self.pipeline.start()
    #Extracts the thermal frame from the output queue
    def extract(self):
        thermalImgPacket = self.thermalImgOut.tryGet() 
        if thermalImgPacket is not None:
            return thermalImgPacket.getCvFrame() 
        else:
            return None
    #Processes the thermal frame and returns the color map
    def process(self, frame):
        if frame is None:
            return None
        return cv2.applyColorMap(frame, cv2.COLORMAP_PLASMA)
    #Displays the color map
    def display(self, frame):
        if frame is not None:
            cv2.imshow('Color Map', frame)
    #Runs the pipeline locally
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
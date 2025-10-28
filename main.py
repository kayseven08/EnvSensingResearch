from microphone import MicrophoneIntegration as Microphone
from optical_camera import Camera
from thermal import Thermal
import numpy as np, matplotlib.pyplot as plt, cv2
class Main:
    def __init__(self):
        self.mic = Microphone()
        self.cam = Camera()
        self.thermal = Thermal()
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(0, self.mic.CHUNK)
        self.line, = self.ax.plot(self.x, np.random.rand(self.mic.CHUNK))
        self.ax.set_ylim(-32768, 32767)
        self.ax.set_xlim(0, self.mic.CHUNK)
        self.ax.set_ylabel("amplitude")
        self.ax.set_xlabel("frames")
        self.ax.set_title("mic adudio strean")
        self.channel_data_buffer = []
        plt.show(block=False)
    def extract(self):
        data, rms = self.mic.extract()
        cam_frame = self.cam.extract()
        thermal_frame = self.thermal.extract()
        return data, rms, cam_frame, thermal_frame

    def process(self, data, rms, cam_frame, thermal_frame):
        channel_data, db  = self.mic.process(data, rms)
        processed_cam_frame = self.cam.process(cam_frame)
        procssed_thermal_frame = self.thermal.process(thermal_frame)
        return channel_data, db, processed_cam_frame, procssed_thermal_frame

    def display(self, channel_data, db, processed_cam_frame, processed_thermal_frame):
        # mic.display(channel_data, db)
        # cam.display(processed_cam_frame)
        # thermal.display(processed_thermal_frame)
        # if(processed_thermal_frame.size)
        cv2_concat = None
        if (processed_thermal_frame is not None ):
            # print(processed_thermal_frame.shape)
            # print(processed_cam_frame.shape)
            cam_frame = cv2.resize(processed_cam_frame, (512, 384))
            thermal_frame = cv2.resize(processed_thermal_frame, (512, 384))
            cv2_concat = cv2.hconcat([cam_frame, thermal_frame])
        else:
            cam_frame = cv2.resize(processed_cam_frame, (512, 384))
            cv2_concat = cv2.hconcat([cam_frame, np.zeros_like(cam_frame)])
        self.channel_data_buffer.append(channel_data)
        print(f"Volume (decibels): {db}")
        self.line.set_ydata(channel_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        cv2.imshow("Camera/Thermal", cv2_concat)

    def main(self):
        while True:
            data, rms, cam_frame, thermal_frame = self.extract()
            channel_data, db, processed_cam_frame, processed_thermal_frame = self.process(data, rms, cam_frame, thermal_frame)
            self.display(channel_data, db, processed_cam_frame, processed_thermal_frame)

if __name__ == "__main__":
    thing = Main()
    thing.main()
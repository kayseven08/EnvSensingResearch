from microphone import MicrophoneIntegration as Microphone
from optical_camera import Camera
from thermal import Thermal
import numpy as np, matplotlib.pyplot as plt, cv2
from datetime import datetime, timedelta
import threading, queue
class Main:
    def __init__(self):
        self.mic_queue = queue.Queue(maxsize=1)
        self.cam_queue = queue.Queue(maxsize=1)
        self.thermal_queue = queue.Queue(maxsize=1)
        def _init_mic():
            mic = Microphone()
            try:
                self.mic_queue.put_nowait(mic)
            except queue.Full:
                pass
        
        def _init_cam():
            cam = Camera()
            try:
                self.cam_queue.put_nowait(cam)
            except queue.Full:
                pass
                
        def _init_thermal():
            thermal = Thermal()
            try:
                self.thermal_queue.put_nowait(thermal)
            except queue.Full:
                pass
        mic_thread = threading.Thread(target=_init_mic)
        cam_thread = threading.Thread(target=_init_cam)
        thermal_thread = threading.Thread(target=_init_thermal)

        mic_thread.start()
        cam_thread.start()
        thermal_thread.start()

        mic_thread.join()
        cam_thread.join()
        thermal_thread.join()

        self.mic = self.mic_queue.get()
        self.cam = self.cam_queue.get()
        self.thermal = self.thermal_queue.get()


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
        self.times = []

        self.data_queue = queue.Queue(maxsize=1)
        self.rms_queue = queue.Queue(maxsize=1)
        self.cam_frame_queue = queue.Queue(maxsize=1)
        self.thermal_frame_queue = queue.Queue(maxsize=1)
    
    def _run_threading(self, sensor_method, output_queue: queue.Queue):
        result = sensor_method()
        try:
            output_queue.put_nowait(result)
        except queue.Full:
            pass

    def extract(self):

        cam_thread = threading.Thread(target=self._run_threading, 
                                     args=(self.cam.extract, self.cam_frame_queue))
        thermal_thread = threading.Thread(target=self._run_threading, 
                                         args=(self.thermal.extract, self.thermal_frame_queue))
        def mic_extract_wrapper():
            data, rms = self.mic.extract()
            try:
                self.data_queue.put_nowait(data)
                self.rms_queue.put_nowait(rms)
            except queue.Full:
                pass
        mic_thread = threading.Thread(target=mic_extract_wrapper)

        self.times.append(datetime.now())
        mic_thread.start()
        cam_thread.start()
        thermal_thread.start()

        mic_thread.join()
        cam_thread.join()
        thermal_thread.join()

        data = self.data_queue.get()
        rms = self.rms_queue.get()
        cam_frame = self.cam_frame_queue.get()
        thermal_frame = self.thermal_frame_queue.get()

        return data, rms, cam_frame, thermal_frame

    def process(self, data, rms, cam_frame, thermal_frame):
        def cam_process_wrapper(frame):
            processed_frame = self.cam.process(frame)
            try:
                self.cam_frame_queue.put_nowait(processed_frame)
            except queue.Full:
                pass

        cam_thread = threading.Thread(target=cam_process_wrapper, args=(cam_frame,))

        cam_thread.start()
        channel_data, db  = self.mic.process(data, rms)
        procssed_thermal_frame = self.thermal.process(thermal_frame)
        cam_thread.join()
        processed_cam_frame = self.cam_frame_queue.get()
        
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
        print(f"Time of extraction: {self.times[-1]}")
        print(f"latency: {(datetime.now() - self.times[-1]).total_seconds()}")

    def add_display(self, channel_data, db, processed_cam_frame, processed_thermal_frame):
        try:
            self.final_data_queue.put(channel_data)
            self.final_db_queue.put(db)
            self.final_cam_frame_queue.put(processed_cam_frame)
            self.final_thermal_frame_queue.put(processed_thermal_frame)
        except Exception as e:
            print(f"errored in adding to queues {e}")
            pass

    def main(self):
        self.final_data_queue = queue.Queue(maxsize=1)
        self.final_db_queue = queue.Queue(maxsize=1)
        self.final_cam_frame_queue = queue.Queue(maxsize=1)
        self.final_thermal_frame_queue = queue.Queue(maxsize=1)

        def mic_display():
            print('started mic display')
            while True:
                channel_data = self.final_data_queue.get()
                db = self.final_db_queue.get()
                self.channel_data_buffer.append(channel_data)
                print(f"Volume (decibels): {db}")
                self.line.set_ydata(channel_data)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        def cam_thermal_display():
            print('started cam/thermal display')
            while True:
                processed_cam_frame = self.final_cam_frame_queue.get()
                processed_thermal_frame = self.final_thermal_frame_queue.get()
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
                cv2.imshow("Camera/Thermal", cv2_concat)
                print(f"Time of extraction: {self.times[-1]}")
                print(f"latency: {(datetime.now() - self.times[-1]).total_seconds()}")

        self.mic_display_thread = threading.Thread(target=mic_display)
        self.cam_thermal_display_thread = threading.Thread(target=cam_thermal_display)
        
        self.mic_display_thread.start()
        self.cam_thermal_display_thread.start()


        while True:
            data, rms, cam_frame, thermal_frame = self.extract()
            channel_data, db, processed_cam_frame, processed_thermal_frame = self.process(data, rms, cam_frame, thermal_frame)
            self.add_display(channel_data, db, processed_cam_frame, processed_thermal_frame)

if __name__ == "__main__":
    thing = Main()
    thing.main()
from microphone import MicrophoneIntegration as Microphone
from optical_camera import Camera
from thermal import Thermal
import numpy as np, matplotlib.pyplot as plt, cv2
from datetime import datetime, timedelta
import threading, queue, time
class Main:
    def __init__(self):
        self.mic_queue = queue.Queue(maxsize=1)
        self.cam_queue = queue.Queue(maxsize=1)
        self.final_data_queue = queue.Queue(maxsize=1)
        self.final_db_queue = queue.Queue(maxsize=1)
        self.final_cam_frame_queue = queue.Queue(maxsize=1)
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
                
        mic_thread = threading.Thread(target=_init_mic)
        cam_thread = threading.Thread(target=_init_cam)

        mic_thread.start()
        cam_thread.start()

        mic_thread.join()
        cam_thread.join()

        self.mic = self.mic_queue.get()
        self.cam = self.cam_queue.get()


        # self.fig, self.ax = plt.subplots()
        # self.x = np.arange(0, self.mic.CHUNK)
        # self.line, = self.ax.plot(self.x, np.random.rand(self.mic.CHUNK))
        # self.ax.set_ylim(-32768, 32767)
        # self.ax.set_xlim(0, self.mic.CHUNK)
        # self.ax.set_ylabel("amplitude")
        # self.ax.set_xlabel("frames")
        # self.ax.set_title("mic adudio strean")

        self.plot_height = 384
        self.plot_width = 512
        # Pre-calculate X-axis points for the plot
        self.plot_x_points = np.linspace(0, self.plot_width - 1, num=self.mic.CHUNK).astype(np.int32)
        # Pre-allocate a black frame to draw on (and for empty thermal)
        self.empty_frame = np.zeros((self.plot_height, self.plot_width, 3), dtype=np.uint8)

        self.channel_data_buffer = []
        plt.show(block=False)
        self.times = []

        self.data_queue = queue.Queue(maxsize=1)
        self.rms_queue = queue.Queue(maxsize=1)
        self.cam_frame_queue = queue.Queue(maxsize=1)

        self.raw_data_all_queue = queue.Queue(maxsize=1)
        self.processed_data_all_queue = queue.Queue(maxsize=1)    
    def _run_threading(self, sensor_method, output_queue: queue.Queue):
        result = sensor_method()
        try:
            output_queue.put_nowait(result)
        except queue.Full:
            pass

    def extract(self):

        cam_thread = threading.Thread(target=self._run_threading, 
                                     args=(self.cam.extract, self.cam_frame_queue))
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

        mic_thread.join()
        cam_thread.join()

        data = self.data_queue.get()
        rms = self.rms_queue.get()
        cam_frame = self.cam_frame_queue.get()
        
        return data, rms, cam_frame,

    def process(self, data, rms, cam_frame):
        def cam_process_wrapper(frame):
            processed_frame = self.cam.process(frame)
            try:
                self.cam_frame_queue.put_nowait(processed_frame)
            except queue.Full:
                pass

        cam_thread = threading.Thread(target=cam_process_wrapper, args=(cam_frame,))

        cam_thread.start()
        channel_data, db  = self.mic.process(data, rms)
        cam_thread.join()
        processed_cam_frame = self.cam_frame_queue.get()
        
        return channel_data, db, processed_cam_frame

    def _extraction_loop(self):
        while True:
            try:
                data, rms, cam_frame = self.extract()
                timestamp = datetime.now()
                self.raw_data_all_queue.put((data, rms, cam_frame, timestamp))
            except Exception as e:
                print(f"error in extraction loop: {e}")
                time.sleep(1)
    
    def _processing_loop(self):
        while True:
            try:
                data, rms, cam_frame, timestamp = self.raw_data_all_queue.get()
                channel_data, db, processed_cam_frame = self.process(data, rms, cam_frame)
                self.processed_data_all_queue.put((channel_data, db, processed_cam_frame, timestamp))
            except Exception as e:
                print(f"error in extraction loop: {e}")
                time.sleep(1)

    def display(self, channel_data, db, processed_cam_frame):
        
        cam_frame = cv2.resize(processed_cam_frame, (self.plot_width, self.plot_height))
        plot_img = self.empty_frame.copy() 

        normalized_data = (channel_data + 32768) / 65536.0
        y_points = (normalized_data * (self.plot_height - 1)).astype(np.int32)
        y_points = self.plot_height - 1 - y_points
        points = np.stack((self.plot_x_points, y_points), axis=-1)
        cv2.polylines(plot_img, [points], isClosed=False, color=(0, 255, 0), thickness=1)

        cv2.putText(plot_img, f"Volume (dB): {db:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2_concat = cv2.hconcat([cam_frame, plot_img])
        cv2.imshow("Camera/Audio", cv2_concat)


    def main(self, display_parallel=False):
        self.extractor_thread = threading.Thread(target=self._extraction_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
        print("Starting pipeline threads...")
        self.extractor_thread.start()
        self.processor_thread.start()
        print("Pipeline started.")

        while True:
            try:
                channel_data, db, processed_cam_frame, timestamp = self.processed_data_all_queue.get()
                predisplay_latency = (datetime.now() - timestamp).total_seconds()
                print(f"Predisplay latency: {predisplay_latency*1000:.2f}")
                self.display(channel_data, db, processed_cam_frame)
                latency = (datetime.now() - timestamp).total_seconds()
                print(f"Total end-to-end latency: {latency*1000:.2f} ms")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping...")
                    break
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                print("stopping")
                break
            except Exception as e:
                print(f"error in display loop: {e}")
        cv2.destroyAllWindows()
if __name__ == "__main__":
    thing = Main()
    thing.main(display_parallel=False)
from microphone import MicrophoneIntegration as Microphone
from optical_camera import Camera
from thermal import Thermal
import numpy as np, matplotlib.pyplot as plt, cv2
from datetime import datetime, timedelta
import threading, queue, time
class Main:
    def __init__(self):
        #Queues that allow for data to pass between threads
        #setting up threads for initializing the sensors (makes it faster)
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

        #Setting up configuration for the audio plotting in OpenCV
        self.plot_height = 384
        self.plot_width = 512
        self.plot_x_points = np.linspace(0, self.plot_width - 1, num=self.mic.CHUNK).astype(np.int32)
        self.empty_frame = np.zeros((self.plot_height, self.plot_width, 3), dtype=np.uint8)

        self.channel_data_buffer = []
        self.times = []

        self.data_queue = queue.Queue(maxsize=1)
        self.rms_queue = queue.Queue(maxsize=1)
        self.cam_frame_queue = queue.Queue(maxsize=1)
        self.thermal_frame_queue = queue.Queue(maxsize=1)

        #Extracted data queue used in pipeline mode
        self.raw_data_all_queue = queue.Queue(maxsize=1)
        #Processed data queue used in pipeline mode
        self.processed_data_all_queue = queue.Queue(maxsize=1)
    #Performs the extraction methods of the sensors in a separate thread
    def _run_threading(self, sensor_method, output_queue: queue.Queue):
        result = sensor_method()
        try:
            output_queue.put_nowait(result)
        except queue.Full:
            pass

    #Creates and runs the threads for extracting the data from the sensors
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

        self.times.append(datetime.now()) #gets current time used to calculate latency
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


    #Creates and runs the threads for processing the data from the sensors
    def process(self, data, rms, cam_frame, thermal_frame):
        def cam_process_wrapper(frame):
            processed_frame = self.cam.process(frame)
            try:
                self.cam_frame_queue.put_nowait(processed_frame)
            except queue.Full:
                pass

        cam_thread = threading.Thread(target=cam_process_wrapper, args=(cam_frame,))

        cam_thread.start()
        channel_data, db  = self.mic.process(data, rms) #not run in parallel because it is quick
        processed_thermal_frame = self.thermal.process(thermal_frame) #not run in parallel because it is quick
        cam_thread.join()
        processed_cam_frame = self.cam_frame_queue.get()
        
        return channel_data, db, processed_cam_frame, processed_thermal_frame

    #Extraction loop that runs on individual thread and pushes data to the raw_data_all_queue
    def _extraction_loop(self):
        while True:
            try:
                data, rms, cam_frame, thermal_frame = self.extract()
                timestamp = datetime.now()
                self.raw_data_all_queue.put((data, rms, cam_frame, thermal_frame, timestamp))
            except Exception as e:
                print(f"error in extraction loop: {e}")
                time.sleep(.1)

    #Processing loop that runs on individual thread and pushes data to the processed_data_all_queue
    def _processing_loop(self):
        while True:
            try:
                data, rms, cam_frame,thermal_frame, timestamp = self.raw_data_all_queue.get()
                channel_data, db, processed_cam_frame, processed_thermal_frame = self.process(data, rms, cam_frame, thermal_frame)
                self.processed_data_all_queue.put((channel_data, db, processed_cam_frame, processed_thermal_frame, timestamp))
            except Exception as e:
                print(f"error in extraction loop: {e}")
                time.sleep(.1)
    #Displays the data in OpenCV
    def display(self, channel_data, db, processed_cam_frame, processed_thermal_frame):
        cam_frame = cv2.resize(processed_cam_frame, (self.plot_width, self.plot_height))
        cv2_concat = cam_frame.copy()
        if (processed_thermal_frame is not None ): #used because thermal frame takes longer to initialize
            print('thermal frame exists')
            thermal_frame = cv2.resize(processed_thermal_frame, (self.plot_width, self.plot_height))
            #Concatenates the thermal frame to the camera frame after it initializes
            cv2_concat = cv2.hconcat([cam_frame, thermal_frame])

        plot_img = self.empty_frame.copy() 
        #Normalizes the data and converts it to an integer for plotting
        normalized_data = (channel_data + 32768) / 65536.0
        y_points = (normalized_data * (self.plot_height - 1)).astype(np.int32)
        y_points = self.plot_height - 1 - y_points
        points = np.stack((self.plot_x_points, y_points), axis=-1)
        #Draws the audio waveform on the plot
        cv2.polylines(plot_img, [points], isClosed=False, color=(0, 255, 0), thickness=1)
        #Writes the volume to the plot
        cv2.putText(plot_img, f"Volume (dB): {db:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #Concatenates the frames and plots them
        cv2_hconcat = cv2.hconcat([cv2_concat, plot_img])
        cv2.imshow("Camera/Thermal/Audio", cv2_hconcat)
        print('displayed')

    def run_pipelines(self):
        self.extractor_thread = threading.Thread(target=self._extraction_loop, daemon=True)
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
        print("Starting pipeline threads...")
        self.extractor_thread.start()
        self.processor_thread.start()
        print("Pipeline started.")

        while True:
            try:
                channel_data, db, processed_cam_frame, thermal_frame, timestamp = self.processed_data_all_queue.get()
                self.display(channel_data, db, processed_cam_frame, thermal_frame)
                latency = (datetime.now() - timestamp).total_seconds()
                print(f"Total end-to-end latency: {latency*1000:.2f} ms")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping...")
                    break
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                print("stopping")
                return
            except Exception as e:
                print(f"error in display loop: {e}")
    def run(self):
        while True:
            data, rms, cam_frame, thermal_frame = self.extract()
            channel_data, db, processed_cam_frame, processed_thermal_frame = self.process(data, rms, cam_frame, thermal_frame)
            self.display(channel_data, db, processed_cam_frame, processed_thermal_frame)
            latency = (datetime.now() - self.times[-1]).total_seconds()
            print(f"Total end-to-end latency: {latency*1000:.2f} ms")

if __name__ == "__main__":
    thing = Main()
    thing.run_pipelines()
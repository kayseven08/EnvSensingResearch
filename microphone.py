import threading
import pyaudio, numpy as np, matplotlib.pyplot as plt, speech_recognition as sr, audioop, math
from matplotlib.animation import FuncAnimation
class MicrophoneIntegration:
    #Initializes pyaudio and audio stream as well as the configuration
    def __init__(self):
        self.CHUNK = 1024
        self.WIDTH = 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 6
        self.RATE = 16000
        self.INDEX = 1
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.p.get_format_from_width(self.WIDTH),
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        input_device_index = self.INDEX,
                        frames_per_buffer=self.CHUNK)

        self.r = sr.Recognizer()
    #Runs to initialize the matplotlib plot when running just this file
    def initialize_local(self):
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(0, self.CHUNK)
        self.line, = self.ax.plot(self.x, np.random.rand(self.CHUNK))
        self.ax.set_ylim(-32768, 32767)
        self.ax.set_xlim(0, self.CHUNK)
        self.ax.set_ylabel("amplitude")
        self.ax.set_xlabel("frames")
        self.ax.set_title("mic audio stream")
        print("Recording")
        self.channel_data_buffer = []
        plt.show(block=False)
    #Extracts the audio data from the stream
    def extract(self):
        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        rms = audioop.rms(data, self.WIDTH) #calculates the rms of the audio from the stream
        return data, rms
    #Processes the audio data and calculates a dB value
    def process(self, data, rms):
        np_data = np.frombuffer(data, dtype=np.int16)
        channel_data = np_data[0::self.CHANNELS]
        db =0
        if rms > 0:
            db = 20 * math.log10(rms / 32767.0) 
        else:
            db=-math.inf
        return  channel_data, db
    #Displays the audio data in matplotlib
    def display(self, channel_data, db):
        self.channel_data_buffer.append(channel_data)
        print(f"Volume (decibels): {db}")
        self.line.set_ydata(channel_data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    #Deprecated
    def waveform(self):
        fig, ax = plt.subplots()
        x = np.arange(0, self.CHUNK)
        line, = ax.plot(x, np.random.rand(self.CHUNK))
        ax.set_ylim(-32768, 32767)
        ax.set_xlim(0, self.CHUNK)
        ax.set_ylabel("amplitude")
        ax.set_xlabel("frames")
        ax.set_title("mic adudio strean")
        print("Recording")
        self.audio_buffer = []
        def recognize_audio(audio_data):
            audio = sr.AudioData(audio_data, self.RATE, self.WIDTH)
            try:
                text = self.r.recognize_google(audio)
                print(f"You said: {text}")
            except sr.UnknownValueError as e:
                print(f"Could not understand audio | {e}")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                print(e)
        def update_plot(frame):
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                rms = audioop.rms(data, self.WIDTH)
                if rms > 0:
                    db = 20 * math.log10(rms / 32767.0) 
                else:
                    db=-math.inf
                print(f"Volume (decibels): {db} | rms: {rms}")
                np_data = np.frombuffer(data, dtype=np.int16)
                channel_data = np_data[0::self.CHANNELS]
                self.audio_buffer.append(data)
                if len(self.audio_buffer) * self.CHUNK / self.RATE > 3:
                    audio_data = b''.join(self.audio_buffer)
                    self.audio_buffer = []
                    # recognize_audio(audio_data)
                line.set_ydata(channel_data)
            except IOError:
                print("audio overflowed; plotting might lag.")
            return line,

        animation = FuncAnimation(fig, update_plot, blit=True, interval=50)

        try:
            plt.show()
        except KeyboardInterrupt:
            print("Keyboard Interrrup")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    mi = MicrophoneIntegration()
    mi.initialize_local()
    while True:
        data, rms = mi.extract()
        channel_data, db = mi.process(data, rms)
        mi.display(channel_data, db)
    mi.close()
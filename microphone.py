import threading
import pyaudio, numpy as np, matplotlib.pyplot as plt, speech_recognition as sr
from matplotlib.animation import FuncAnimation
class MicrophoneIntegration:
    def __init__(self):
        self.CHUNK = 1024
        self.WIDTH = 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 6
        self.RATE = 16000
        self.INDEX = 0
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.p.get_format_from_width(self.WIDTH),
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        input_device_index = self.INDEX,
                        frames_per_buffer=self.CHUNK)

        self.r = sr.Recognizer()



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
                np_data = np.frombuffer(data, dtype=np.int16)
                channel_data = np_data[0::self.CHANNELS]
                self.audio_buffer.append(data)
                if len(self.audio_buffer) * self.CHUNK / self.RATE > 3:
                    audio_data = b''.join(self.audio_buffer)
                    self.audio_buffer = []
                    recognize_audio(audio_data)
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

microphone_int = MicrophoneIntegration()
microphone_int.waveform()
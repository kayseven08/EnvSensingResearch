import threading
import pyaudio, numpy as np, matplotlib.pyplot as plt, speech_recognition as sr
from matplotlib.animation import FuncAnimation
def waveform():
    CHUNK = 1024
    WIDTH = 2
    FORMAT = pyaudio.paInt16
    CHANNELS = 6
    RATE = 16000
    INDEX = 0
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index = INDEX,
                    frames_per_buffer=CHUNK)

    r = sr.Recognizer()

    fig, ax = plt.subplots()
    x = np.arange(0, CHUNK)
    line, = ax.plot(x, np.random.rand(CHUNK))
    ax.set_ylim(-32768, 32767)
    ax.set_xlim(0, CHUNK)
    ax.set_ylabel("amplitude")
    ax.set_xlabel("frames")
    ax.set_title("mic adudio strean")
    print("Recording")
    audio_buffer = []
    def recognize_audio(audio_data):
        audio = sr.AudioData(audio_data, RATE, WIDTH)
        try:
            text = r.recognize_google(audio)
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    def update_plot(frame):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            np_data = np.frombuffer(data, dtype=np.int16)
            channel_data = np_data[0::CHANNELS]
            audio_buffer.append(channel_data)
            if len(audio_buffer) * CHUNK / RATE > 3:
                audio_data = b''.join(audio_buffer)
                audio_buffer = []
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
        stream.stop_stream()
        stream.close()
        p.terminate()
waveform()
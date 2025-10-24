from microphone import MicrophoneIntegration as Microphone
from optical_camera import Camera
from thermal import Thermal

def extract(mic: Microphone, cam: Camera, thermal: Thermal):
    data, rms = mic.extract()
    cam_frame = cam.extract()
    thermal_frame = thermal.extract()
    return data, rms, cam_frame, thermal_frame

def process(mic: Microphone, cam: Camera, thermal: Thermal, data, rms, cam_frame, thermal_frame):
    channel_data, db  = mic.process(data, rms)
    processed_cam_frame = cam.process(cam_frame)
    procssed_thermal_frame = thermal.process(thermal_frame)
    return channel_data, db, processed_cam_frame, procssed_thermal_frame

def display(mic: Microphone, cam: Camera, thermal: Thermal, channel_data, db, processed_cam_frame, processed_thermal_frame):
    mic.display(channel_data, db)
    cam.display(processed_cam_frame)
    thermal.display(processed_thermal_frame)

def main():
    mic = Microphone()
    cam = Camera()
    thermal = Thermal()

    while True:
        data, rms, cam_frame, thermal_frame = extract(mic, cam, thermal)
        channel_data, db, processed_cam_frame, processed_thermal_frame = process(mic, cam, thermal, data, rms, cam_frame, thermal_frame)
        display(mic, cam, thermal, channel_data, db, processed_cam_frame, processed_thermal_frame)
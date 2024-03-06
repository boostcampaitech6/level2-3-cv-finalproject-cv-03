import threading
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from arch import *


class StopFlag:
    def __init__(self):
        self.flag = False

    def set(self, value):
        self.flag = value

    def get(self):
        return self.flag


def capture_frames(video_source, buffer, stop_flag, interval=0.1):
    video = cv2.VideoCapture(video_source)

    interval = 0.1
    start_time = time.time()
    next_time = start_time + interval

    while not stop_flag.get():
        success, frame = video.read()

        if not success:
            break

        cur_time = time.time()
        if cur_time >= next_time:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(frame)

            next_time += interval

    video.release()


def predict(model, buffer, stop_flag, frame_per_sec=10, total_sec=3):
    total_time, pred_time, cnt = 0, 0, 0
    init_st = time.time()

    frame_per_pred = frame_per_sec * total_sec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    while not stop_flag.get():
        if len(buffer) >= frame_per_pred:
            total_st = time.time()

            with threading.Lock():
                _frames = [buffer[i] for i in range(frame_per_pred)]
            
            frames = []
            for frame in _frames:
                frame = A.Resize(224, 224)(image=frame)["image"]
                frame = (frame / 255.0).astype(np.float32)
                frames.append(frame)

            frames = np.transpose(np.array(frames), (0, 3, 1, 2))
            frames = torch.from_numpy(frames).unsqueeze(dim=0).to(device)

            pred_st = time.time()

            output = model(frames)
            output = F.softmax(output, dim=1)
            prob, pred = torch.max(output, dim=1)

            pred_time += time.time() - pred_st

            pred_class = "Normal" if pred == 0 else "Shoplifting"
            print(f"class: {pred_class:>11} | probability: {prob.item():.4f} | time: {time.time() - init_st:.2f}s")

            with threading.Lock():
                del buffer[:frame_per_sec]

            total_time += time.time() - total_st
            cnt += 1
            if cnt % 50 == 0:
                print(f"({cnt} iter) total inference time per iter: {total_time / cnt}")
                print(f"({cnt} iter) just prediction time per iter: { pred_time / cnt}")


def inference(video_source, capture_interval=0.1, frame_per_sec=10, total_sec=3):
    buffer = []
    stop_flag = StopFlag() 

    # define model
    model = ...

    capture_thread = threading.Thread(
        target=capture_frames, args=(video_source, buffer, stop_flag, capture_interval)
    )
    total_timehread = threading.Thread(
        target=predict, args=(model, buffer, stop_flag, frame_per_sec, total_sec))

    capture_thread.start()
    total_timehread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

        stop_flag.set(True)
        capture_thread.join()
        total_timehread.join()


if __name__ == "__main__":
    video_source = "rtsp://10.28.224.201:30576/stream"

    inference(video_source)

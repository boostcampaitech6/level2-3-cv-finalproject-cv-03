import cv2
import torch
import queue
import threading
import time
from arch import MobileNetGRU
import albumentations as A
import numpy as np


frame_queue = queue.Queue(maxsize=1000)
stop_flag = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def capture_frames(video_source, frame_height=224, frame_width=224):
    global stop_flag

    video = cv2.VideoCapture(video_source)

    interval = 0.05
    start_time = time.time()
    next_time = start_time + interval

    while not stop_flag:
        success, frame = video.read()

        if not success:
            break

        cur_time = time.time()
        if cur_time >= next_time:
            if frame_queue.full():
                pass
            else:
                frame = A.Resize(frame_height, frame_width)(image=frame)[
                    "image"
                ]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put(frame)

            next_time += interval

    video.release()


def inference(frame_per_sec=10):
    global stop_flag

    frame_per_inference = frame_per_sec * 3

    model = MobileNetGRU()
    model = model.to(device)
    model.eval()

    while not stop_flag:
        if frame_queue.qsize() >= frame_per_inference:
            frames = [frame_queue.get() for _ in range(frame_per_inference)]
            frames = np.transpose(np.array(frames), (0, 3, 1, 2))
            frames = (frames / 255.0).astype(np.float32)
            frames = torch.from_numpy(frames).unsqueeze(dim=0).to(device)

            output = model(frames)

            prob, pred = torch.max(output, dim=1)
            pred_class = "Normal" if pred == 0 else "Shoplifting"
            print(f"class: {pred_class:<11} | probability: {prob.item():.4f}")

            for _ in range(frame_per_sec):
                frame_queue.get()


def main(video_source):
    global stop_flag

    capture_thread = threading.Thread(
        target=capture_frames, args=(video_source,)
    )
    inference_thread = threading.Thread(target=inference)

    capture_thread.start()
    inference_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

        stop_flag = True
        capture_thread.join()
        inference_thread.join()


if __name__ == "__main__":
    video_source = ""

    main(video_source)

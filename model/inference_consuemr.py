import os
import json
import redis
import threading
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from arch import MobileNetGRU


class StopFlag:
    def __init__(self):
        self.flag = False

    def set(self, value):
        self.flag = value

    def get(self):
        return self.flag


def make_video_file(output_path, frames):
    global fps
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def capture_frames(
    cctv_info,
    buffer,
    stop_flag,
    interval=0.1,
    save_interval=2 * 60 * 60,
    max_save_time=5 * 60,
):
    global temp_frames, fps, output_dir
    video = cv2.VideoCapture(cctv_info["cctv_url"])

    fps = video.get(cv2.CAP_PROP_FPS)
    max_frames = max_save_time * fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(
        output_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    next_time = start_time + interval
    save_time = start_time + save_interval

    while not stop_flag.get() and video.isOpened():
        success, frame = video.read()
        if not success:
            break

        video_writer.write(frame)
        temp_frames.append(frame)

        cur_time = time.time()
        if cur_time >= next_time:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(frame)
            next_time += interval

        if cur_time >= save_time:
            video_writer.release()
            output_path = os.path.join(
                output_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )
            save_time += save_interval

        if len(temp_frames) > max_frames:
            del temp_frames[0]

    video_writer.release()
    video.release()


def predict(cctv_id, model, buffer, stop_flag, frame_per_sec=10, total_sec=3):
    total_time, pred_time, cnt = 0, 0, 0
    pred_queue, prob_queue = [], []
    queue_limit, anomaly_limit = 5, 3
    global threshold, save_time_length
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
            prob = prob.item()

            pred_time += time.time() - pred_st

            pred_class = "Normal" if pred == 0 else "Shoplifting"
            print(
                f"class: {pred_class:>11} | probability: {prob:.4f} | time: {time.time() - init_st:.2f}s | threshold: {threshold} | save_time_length: {save_time_length}"
            )

            pred_queue.append(pred)
            prob_queue.append(prob if pred == 1 else 0)
            if len(pred_queue) >= queue_limit:
                if (
                    sum(pred_queue) >= anomaly_limit
                    and sum(prob_queue) / sum(pred_queue) > threshold
                ):
                    anomaly_score = float(sum(prob_queue) / sum(pred_queue))
                    print(f"anomaly create.. score : {anomaly_score}")
                    redis_server.lpush(f"{cctv_id}_anomaly", anomaly_score)
                    pred_queue.clear()
                    prob_queue.clear()

                if len(pred_queue) > 0:
                    del pred_queue[0]
                    del prob_queue[0]

            with threading.Lock():
                del buffer[:frame_per_sec]

            total_time += time.time() - total_st
            cnt += 1
            if cnt % 50 == 0:
                print(
                    f"({cnt} iter) total inference time per iter: {total_time / cnt}"
                )
                print(
                    f"({cnt} iter) just prediction time per iter: {pred_time / cnt}"
                )


def inference(cctv_info, capture_interval=0.1, frame_per_sec=10, total_sec=3):

    buffer = []
    stop_flag = StopFlag()

    # define model
    model = MobileNetGRU()

    capture_thread = threading.Thread(
        target=capture_frames,
        args=(cctv_info, buffer, stop_flag, capture_interval),
    )
    predict_thread = threading.Thread(
        target=predict,
        args=(
            cctv_info["cctv_id"],
            model,
            buffer,
            stop_flag,
            frame_per_sec,
            total_sec,
        ),
    )

    capture_thread.start()
    predict_thread.start()

    global threshold, save_time_length, temp_frames, output_dir
    threshold, save_time_length, temp_frames = (
        cctv_info["threshold"],
        cctv_info["save_time_length"],
        [],
    )
    output_dir = f"/data/saved_video/{cctv_info['cctv_id']}"

    flag_key = f"{cctv_info['cctv_id']}_stop_inf"
    alarm_key = f"{cctv_info['cctv_id']}_alarm"
    anomaly_key = f"{cctv_info['cctv_id']}_anomaly"
    while True:
        k, v = redis_server.brpop(
            keys=[flag_key, alarm_key, anomaly_key], timeout=None
        )
        k, v = k.decode("utf-8"), v.decode("utf-8")
        if k == flag_key:
            break
        elif k == alarm_key:
            alarm_config = json.loads(v)
            threshold = alarm_config["threshold"]
            save_time_length = alarm_config["save_time_length"]
        elif k == anomaly_key:
            # anomaly_score = v
            anomaly_save_path = f"{output_dir}/temp_videos/{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            if not os.path.isdir(os.path.dirname(anomaly_save_path)):
                os.makedirs(os.path.dirname(anomaly_save_path), exist_ok=True)
            print(f"make_video_file : {len(temp_frames)}/{anomaly_save_path}")
            make_video_file(anomaly_save_path, temp_frames)

    stop_flag.set(True)
    capture_thread.join()
    predict_thread.join()
    print("stop inference :", flag_key)


if __name__ == "__main__":
    redis_server = redis.Redis(host="10.28.224.201", port=30575, db=0)
    while True:
        key, value = redis_server.brpop(keys="start_inf", timeout=None)
        cctv_info = json.loads(value.decode("utf-8"))
        print("start inferene :", cctv_info)
        inference(cctv_info)

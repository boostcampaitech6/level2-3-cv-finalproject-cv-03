import os
import glob
import json
import redis
import requests
import subprocess
import threading
import time
import uuid
import shutil
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from arch import *
from utils import RNN_INPUT_SIZE


class StopFlag:
    def __init__(self):
        self.flag = False

    def set(self, value):
        self.flag = value

    def get(self):
        return self.flag


def get_model():
    class MobileNet(nn.Module):
        def __init__(self):
            super(MobileNet, self).__init__()
            self.features = models.mobilenet_v2(weights="DEFAULT").features
            self.avgpool = nn.AvgPool2d((20, 20))

        def __str__(self):
            return "MobileNet"

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            return x

    cnn = MobileNet()
    input_size = RNN_INPUT_SIZE[str(cnn)]
    rnn = GRU(input_size=input_size, hidden_size=256, num_layers=1)
    model = ClipModel(cnn, rnn, hidden=256, nc=3)
    return model


def save_anomaly_log(cctv_id, anomaly_score):
    global fps, save_time_length, save_image_dir, save_hls_log_video_dir

    current_time = time.strftime("%Y%m%d_%H%M%S")
    anomaly_save_path = f"{save_hls_log_video_dir}/{current_time}.mp4"
    anomaly_temp_dir = f"{save_hls_log_video_dir}/temp/{uuid.uuid4()}"
    os.makedirs(anomaly_temp_dir, exist_ok=True)

    save_image_cnt = int(fps * save_time_length * 60)
    save_image_paths = sorted(glob.glob(f"{save_image_dir}/*"))
    save_image_paths = save_image_paths[-save_image_cnt:]
    print("save_image_paths :", len(save_image_paths))

    for save_image_path in save_image_paths:
        shutil.copy(
            save_image_path,
            os.path.join(anomaly_temp_dir, os.path.basename(save_image_path)),
        )

    command = [
        "ffmpeg",
        "-pattern_type",
        "glob",
        "-r",
        str(fps),
        "-i",
        f"{anomaly_temp_dir}/*.png",
        "-loglevel",
        "panic",
        anomaly_save_path,
    ]
    subprocess.run(command, check=True)

    shutil.rmtree(anomaly_temp_dir)

    with open(anomaly_save_path, "rb") as video_file:
        files = {"video_file": (anomaly_save_path, video_file)}
        requests.post(
            f"http://10.28.224.201:30438/api/v0/cctv/log_register?cctv_id={cctv_id}&anomaly_create_time={current_time}&anomaly_score={anomaly_score}&anomaly_save_path={anomaly_save_path}",
            files=files,
        )


def save_video(image_paths):
    global fps, save_video_dir
    frame = cv2.imread(image_paths[0])
    height, width = frame.shape[:2]
    video_name = os.path.basename(image_paths[0]).split(".")[0] + ".mp4"
    video_path = os.path.join(save_video_dir, video_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image_path in image_paths:
        video.write(cv2.imread(image_path))
        os.remove(image_path)
    video.release()


def capture_frames(
    cctv_info,
    buffer,
    stop_flag,
    interval=0.1,
    save_interval=2 * 60 * 60,
    max_save_time=5 * 60,
):
    global fps, save_image_dir, save_video_dir
    video = cv2.VideoCapture(cctv_info["cctv_url"])

    fps = video.get(cv2.CAP_PROP_FPS)
    max_frames = max_save_time * fps

    ## Test Setting!
    # max_save_time = 0
    # save_interval = 10
    # max_frames = 10

    start_time = time.time()
    next_time = start_time + interval
    save_time = start_time + save_interval + max_save_time

    while not stop_flag.get() and video.isOpened():
        success, frame = video.read()
        if not success:
            break

        cur_time = time.time()
        if cur_time >= next_time:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resize_frame = A.Resize(224, 224)(image=frame)["image"]
            norm_frame = (resize_frame / 255.0).astype(np.float32)

            image_path = os.path.join(
                save_image_dir,
                f"{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}.png",
            )
            cv2.imwrite(
                image_path, cv2.cvtColor(resize_frame, cv2.COLOR_RGB2BGR)
            )
            buffer.append(norm_frame)
            next_time += interval

        if cur_time >= save_time:
            save_time += save_interval
            image_paths = sorted(glob.glob(f"{save_image_dir}/*.png"))
            image_paths = image_paths[:-max_frames]
            # redis로 2시간 스케줄링 진행
            redis_server.lpush(
                f"{cctv_info['cctv_id']}_save_video",
                json.dumps({"image_paths": image_paths}),
            )

    video.release()


def predict(cctv_id, model, buffer, stop_flag, frame_per_sec=10, total_sec=3):
    total_time, pred_time, cnt = 0, 0, 0
    sleep_time, sleep_cnt, sleep_flag = 5, 0, False

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
                frames = [buffer[i] for i in range(frame_per_pred)]

            with threading.Lock():
                del buffer[:frame_per_sec]

            if sleep_flag:
                if sleep_time > sleep_cnt:
                    sleep_cnt += 1
                    continue
                else:
                    redis_server.lpush(f"{cctv_id}_anomaly", anomaly_score)

                    sleep_cnt = 0
                    sleep_flag = False

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

            if pred and prob > threshold:
                sleep_flag = True
                anomaly_score = float(prob)  # noqa: F841

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
    # model = get_model()
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

    global threshold, save_time_length, save_image_dir, save_video_dir, save_hls_log_video_dir
    threshold, save_time_length = (
        cctv_info["threshold"],
        cctv_info["save_time_length"],
    )

    save_hls_log_video_dir = (
        f"/data/saved/saved_log_videos/{cctv_info['cctv_id']}"
    )
    save_image_dir = f"/data/saved/saved_images/{cctv_info['cctv_id']}"
    save_video_dir = f"/data/saved/saved_videos/{cctv_info['cctv_id']}"

    os.makedirs(save_hls_log_video_dir, exist_ok=True)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_video_dir, exist_ok=True)

    flag_key = f"{cctv_info['cctv_id']}_stop_inf"
    alarm_key = f"{cctv_info['cctv_id']}_alarm"
    anomaly_key = f"{cctv_info['cctv_id']}_anomaly"
    save_video_key = f"{cctv_info['cctv_id']}_save_video"
    while True:
        k, v = redis_server.brpop(
            keys=[flag_key, alarm_key, anomaly_key, save_video_key],
            timeout=None,
        )
        k, v = k.decode("utf-8"), v.decode("utf-8")
        if k == flag_key:
            print("stop inference :", flag_key)
            break
        elif k == alarm_key:
            alarm_config = json.loads(v)
            print(
                f'change threshold : {threshold} -> {alarm_config["threshold"]}'
            )
            print(
                f'change save_time_length : {save_time_length} -> {alarm_config["save_time_length"]}'
            )
            threshold = alarm_config["threshold"]
            save_time_length = alarm_config["save_time_length"]
        elif k == save_video_key:
            print("saving video..")
            image_paths = json.loads(v)["image_paths"]
            save_video(image_paths)
        elif k == anomaly_key:
            print(f"anomaly create.. score : {v}")
            anomaly_score = v
            save_anomaly_log(cctv_info["cctv_id"], anomaly_score)

    stop_flag.set(True)
    capture_thread.join()
    predict_thread.join()


if __name__ == "__main__":
    redis_server = redis.Redis(host="10.28.224.201", port=30435, db=0)
    while True:
        key, value = redis_server.brpop(keys="start_inf", timeout=None)
        cctv_info = json.loads(value.decode("utf-8"))
        print("start inferene :", cctv_info)
        inference(cctv_info)

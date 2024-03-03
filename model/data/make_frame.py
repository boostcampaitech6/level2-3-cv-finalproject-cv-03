import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A


def make_frames(
    video_path, save_dir_path, frame_num, frame_height, frame_width
):
    video = cv2.VideoCapture(video_path)
    video_frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(int(round(video_frame_num / frame_num)), 1)

    frames = []
    cur_idx = 0

    for _ in range(frame_num):
        video.set(cv2.CAP_PROP_POS_FRAMES, cur_idx)
        success, frame = video.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = A.Resize(frame_height, frame_width)(image=frame)["image"]
        frames.append(frame)

        cur_idx = min(cur_idx + interval, video_frame_num - 1)

    video.release()

    frames = np.transpose(np.array(frames), (0, 3, 1, 2))

    file_name = os.path.splitext(os.path.basename(video_path))[0]
    np.save(os.path.join(save_dir_path, f"{file_name}.npy"), frames)


def main(video_dir_path, save_dir_path, frame_num, frame_height, frame_width):
    video_paths = [
        os.path.join(video_dir_path, fname)
        for fname in os.listdir(video_dir_path)
    ]

    for video_path in tqdm(video_paths):
        make_frames(
            video_path, save_dir_path, frame_num, frame_height, frame_width
        )


if __name__ == "__main__":
    video_dir_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train_clips"
    save_dir_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train_frames"

    frame_num = 24
    frame_height = 224
    frame_width = 224

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    main(video_dir_path, save_dir_path, frame_num, frame_height, frame_width)

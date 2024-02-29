import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self, frame_num, video_dir_path, anno_csv_path, transforms=None
    ):
        csv_data = pd.read_csv(anno_csv_path)

        self.video_paths = [
            os.path.join(video_dir_path, fname)
            for fname in csv_data["clip_fname"]
        ]
        self.video_labels = [
            fname.split("_")[0] for fname in csv_data["clip_fname"]
        ]
        self.frame_num = frame_num
        self.transforms = transforms

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, item):
        frames = []

        video_path = self.video_paths[item]
        video = cv2.VideoCapture(video_path)
        video_frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(round(int(video_frame_num / self.frame_num)), 1)

        for frame_idx in range(
            0, min(video_frame_num, self.frame_num * interval), interval
        ):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()

            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized_frame = frame / 255.0
            normalized_frame = normalized_frame.astype(np.float32)
            frames.append(normalized_frame)

        video.release()

        while len(frames) < self.frame_num:
            if video_frame_num < self.frame_num:
                frames.append(frames[-1])

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))

        labels = [self.video_labels[item] for _ in range(self.frame_num)]

        return frames, labels


frame_num = 64
video_dir_path = "../dataset/videos"
anno_csv_path = "../dataset/metadata.csv"

dataset = TrainDataset(frame_num, video_dir_path, anno_csv_path)

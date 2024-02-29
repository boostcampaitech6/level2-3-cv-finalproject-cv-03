import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, frame_num, video_path, anno_csv_path, transforms=None):
        csv_data = pd.read_csv(anno_csv_path)

        self.video = [
            os.path.join(video_path, i) for i in csv_data["clip_fname"]
        ]
        self.label = [
            i.apply(lambda x: x.split("_")[0]) for i in csv_data["clip_fname"]
        ]
        self.frame_num = frame_num
        self.transforms = transforms

    def __len__(self):
        return len(self.video)

    def __getitem__(self, item):
        frames = []
        labels = []

        video_path = self.video[item]
        video_read = cv2.VideoCapture(video_path)
        video_frame = int(video_read.get(cv2.CAP_PROP_FRAME_COUNT))
        cut_frame = max(int(video_frame / self.frame_num), 1)

        for frame_idx in range(frame_num, video_frame, cut_frame):
            video_read.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_read.read()

            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized_frame = frame / 255.0
            normalized_frame = normalized_frame.astype(np.float32)
            frames.append(normalized_frame)

        video_read.release()

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))

        labels = [self.label[item] for _ in range(frame_num)]

        return frames, labels, video_path


frame_num = 64
video_path = "../dataset/videos"
anno_csv_path = "../dataset/metadata.csv"

dataset = TrainDataset(frame_num, video_path, anno_csv_path)

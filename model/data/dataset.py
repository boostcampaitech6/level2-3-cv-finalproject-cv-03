import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, frame_num, video_dir_path, transforms=None):
        self.video_paths = [
            os.path.join(video_dir_path, fname)
            for fname in os.listdir(video_dir_path)
        ]
        self.video_labels = [
            0 if fname.split("_")[0] == "Normal" else 1
            for fname in os.listdir(video_dir_path)
        ]
        self.frame_num = frame_num
        self.transforms = transforms

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, item):
        frames = []
        cur_idx = 0

        video_path = self.video_paths[item]
        video = cv2.VideoCapture(video_path)
        video_frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(int(round(video_frame_num / self.frame_num)), 1)

        for _ in range(self.frame_num):
            video.set(cv2.CAP_PROP_POS_FRAMES, cur_idx)
            success, frame = video.read()

            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized_frame = (frame / 255.0).astype(np.float32)

            if self.transforms is not None:
                transformed_frame = self.transforms(image=normalized_frame)[
                    "image"
                ]
                frames.append(transformed_frame)
            else:
                frames.append(normalized_frame)

            cur_idx = min(cur_idx + interval, video_frame_num - 1)

        video.release()

        frames = np.array(frames)
        frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)))

        labels = [self.video_labels[item] for _ in range(self.frame_num)]
        labels = torch.tensor(labels, dtype=torch.long)

        return frames, labels


class FrameTrainDataset(Dataset):
    def __init__(self, frame_dir_path, transforms=None):
        self.frame_paths = [
            os.path.join(frame_dir_path, fname)
            for fname in os.listdir(frame_dir_path)
        ]
        self.frame_labels = [
            0 if fname.split("_")[0] == "Normal" else 1
            for fname in os.listdir(frame_dir_path)
        ]
        self.transforms = transforms

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, item):
        frames = np.load(self.frame_paths[item])
        frames = (frames / 255.0).astype(np.float32)

        if self.transforms is not None:
            frames = self.transforms(image=frames)["image"]

        frames = torch.from_numpy(frames)

        labels = [self.frame_labels[item] for _ in range(frames.shape[0])]
        labels = torch.tensor(labels, dtype=torch.long)

        return frames, labels


class ClipTrainDataset(Dataset):
    def __init__(self, frame_dir_path, anno_csv_path, transforms=None):
        self.frame_dir_path = frame_dir_path
        self.anno_df = pd.read_csv(anno_csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.anno_df)

    def __getitem__(self, item):
        annotation = self.anno_df.iloc[item]
        file_name = annotation["file_name"]

        frames = np.load(os.path.join(self.frame_dir_path, file_name))
        frames = (frames / 255.0).astype(np.float32)

        if self.transforms is not None:
            frames = self.transforms(image=frames)["image"]

        frames = torch.from_numpy(frames)

        label = annotation["class"]
        label = torch.tensor(label, dtype=torch.long)

        return frames, label


class ClipValidDataset(Dataset):
    def __init__(self, frame_dir_path, anno_csv_path, transforms=None):
        self.frame_dir_path = frame_dir_path
        self.anno_df = pd.read_csv(anno_csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.anno_df)

    def __getitem__(self, item):
        annotation = self.anno_df.iloc[item]
        file_name = annotation["file_name"]

        frames = np.load(os.path.join(self.frame_dir_path, file_name))
        frames = (frames / 255.0).astype(np.float32)

        if self.transforms is not None:
            frames = self.transforms(image=frames)["image"]

        frames = torch.from_numpy(frames)

        label = annotation["class"]
        label = torch.tensor(label, dtype=torch.long)

        return frames, label, file_name

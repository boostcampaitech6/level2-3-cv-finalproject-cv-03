import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClipTrainDataset(Dataset):
    def __init__(self, clip_dir_path, clip_anno_path, transform=None):
        self.clip_dir_path = clip_dir_path
        self.clip_anno_df = pd.read_csv(clip_anno_path)
        self.transforms = transform

    def __len__(self):
        return len(self.clip_anno_df)

    def __getitem__(self, idx):
        clip_anno = self.clip_anno_df.iloc[idx]
        clip_name = clip_anno["clip_name"]

        frames = np.load(os.path.join(self.clip_dir_path, clip_name))

        if self.transforms is not None:
            transformed_frames = []

            for frame in frames:
                frame = self.transforms(image=frame)["image"]
                transformed_frames.append(frame)

            frames = np.array(transformed_frames)

        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = torch.from_numpy(frames)

        label = clip_anno["class"]
        label = torch.tensor(label, dtype=torch.long)

        return frames, label


class ClipValidDataset(Dataset):
    def __init__(self, clip_dir_path, clip_anno_path, transform=None):
        self.clip_dir_path = clip_dir_path
        self.clip_anno_df = pd.read_csv(clip_anno_path)
        self.transforms = transform

    def __len__(self):
        return len(self.clip_anno_df)

    def __getitem__(self, idx):
        clip_anno = self.clip_anno_df.iloc[idx]
        clip_name = clip_anno["clip_name"]

        frames = np.load(os.path.join(self.clip_dir_path, clip_name))

        if self.transforms is not None:
            transformed_frames = []

            for frame in frames:
                frame = self.transforms(image=frame)["image"]
                transformed_frames.append(frame)

            frames = np.array(transformed_frames)

        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = torch.from_numpy(frames)

        label = clip_anno["class"]
        label = torch.tensor(label, dtype=torch.long)

        return frames, label, clip_name


class GradCAMDataset(Dataset):
    def __init__(self, frames):
        frames = np.transpose(np.array(frames), (0, 3, 1, 2))
        # frames = (frames / 255.0).astype(np.float32)
        self.frames = torch.from_numpy(frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


class FrameTrainDataset(Dataset):
    def __init__(self, clip_dir_path, transforms=None):
        self.frame_paths = [
            os.path.join(clip_dir_path, fname)
            for fname in os.listdir(clip_dir_path)
        ]
        self.frame_labels = [
            0 if fname.split("_")[0] == "Normal" else 1
            for fname in os.listdir(clip_dir_path)
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

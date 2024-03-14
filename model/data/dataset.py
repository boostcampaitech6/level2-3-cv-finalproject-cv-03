import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClipTrainDataset(Dataset):
    def __init__(self, clip_dir_path, anno_clip_path, transforms=None):
        self.clip_dir_path = clip_dir_path
        self.anno_clip = pd.read_csv(anno_clip_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.anno_clip)

    def __getitem__(self, idx):
        anno_clip = self.anno_clip.iloc[idx]
        clip_name = anno_clip["clip_name"]

        frames = np.load(os.path.join(self.clip_dir_path, clip_name))
        frames = (frames / 255.0).astype(np.float32)

        if self.transforms is not None:
            frames = self.transforms(image=frames)["image"]

        frames = torch.from_numpy(frames)

        label = anno_clip["class"]
        label = torch.tensor(label, dtype=torch.long)

        return frames, label


class ClipValidDataset(Dataset):
    def __init__(self, clip_dir_path, anno_clip_path, transforms=None):
        self.clip_dir_path = clip_dir_path
        self.anno_clip = pd.read_csv(anno_clip_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.anno_clip)

    def __getitem__(self, idx):
        anno_clip = self.anno_clip.iloc[idx]
        clip_name = anno_clip["clip_name"]

        frames = np.load(os.path.join(self.clip_dir_path, clip_name))
        frames = (frames / 255.0).astype(np.float32)

        if self.transforms is not None:
            frames = self.transforms(image=frames)["image"]

        frames = torch.from_numpy(frames)

        label = anno_clip["class"]
        label = torch.tensor(label, dtype=torch.long)

        return frames, label, clip_name


class GradCAMDataset(Dataset):
    def __init__(self, frames):
        frames = np.transpose(np.array(frames), (0, 3, 1, 2))
        frames = (frames / 255.0).astype(np.float32)
        self.frames = torch.from_numpy(frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx].unsqueeze(0)


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

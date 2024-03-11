import os
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

from arch import *
from data import *
from trainer import *
from utils import *


def predict(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_dataset = ClipValidDataset(
        args["clip_dir_path"], args["anno_clip_path"]
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=8
    )

    label_tracker = LabelTracker()

    model.load_state_dict(torch.load(args["model_path"]))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for frames, labels, clip_names in tqdm(
            valid_loader, desc="[Valid Prediction]"
        ):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames)
            _, preds = torch.max(outputs, 1)

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()

            label_tracker.update(clip_names=clip_names, labels=preds)

    return label_tracker.result()


def visualization(args, model):
    result = predict(args, model)

    anno_df = pd.read_csv(args["anno_clip_path"])

    for video_name in os.listdir(args["video_dir_path"]):
        file_df = anno_df[anno_df["video_name"] == video_name]
        file_df = file_df.sort_values(by="pred_t")

        video_path = os.path.join(args["video_dir_path"], video_name)
        video = cv2.VideoCapture(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        frames = []
        for _, row in file_df.iterrows():
            idx = row["pred_t"] * fps - 1
            gt_label = row["class"]
            pred_label = result[row["clip_name"]]

            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()

            if not success:
                break

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"true: {CLASSES[gt_label]} | pred: {CLASSES[pred_label]}"

            cv2.putText(
                frame, text, (10, 50), font, 2, (0, 0, 255), 4, cv2.LINE_AA
            )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        video.release()

        save_video_path = os.path.join(args["save_dir_path"], video_name)
        new_video = ImageSequenceClip(frames, fps=2)
        new_video.write_videofile(
            save_video_path, codec="libx264", logger=None
        )


if __name__ == "__main__":
    args = {
        "model_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/save/...pth",
        "clip_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/clips/...",
        "anno_clip_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/anno_clip_val_.csv",
        "video_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/videos",
        "save_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/results",
    }

    cnn = MobileNet()
    rnn = GRU(
        input_size=RNN_INPUT_SIZE[str(cnn)], hidden_size=512, num_layers=1
    )
    model = ClipModel(cnn, rnn, hidden=512, nc=4)

    visualization(args, model)

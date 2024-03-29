import os
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip, concatenate_videoclips
from tqdm import tqdm
import albumentations as A

from data import *
from arch import *
from trainer import *
from utils import *


def predict(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = A.Normalize()

    valid_dataset = ClipValidDataset(
        args["clip_dir_path"], args["clip_anno_path"], transform
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=4, shuffle=False, num_workers=8
    )

    label_tracker = LabelTracker()

    model.load_state_dict(torch.load(args["model_path"]))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for frames, labels, clip_names in tqdm(
            valid_loader, desc="[Predict valid dataset]"
        ):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames)
            outputs = F.softmax(outputs, dim=1)
            probs, preds = torch.max(outputs, 1)

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            probs = probs.cpu().numpy()

            label_tracker.update(
                clip_names=clip_names, labels=preds, probs=probs
            )

    return label_tracker.result()


def visualize(args, labels, probs):
    clips_df = pd.read_csv(args["clip_anno_path"])
    videos = []

    for video_name in tqdm(
        os.listdir(args["video_dir_path"]), desc="[Create prediction videos]"
    ):
        clip_df = clips_df[clips_df["video_name"] == video_name]
        clip_df = clip_df.sort_values(by="pred_frame")

        video_path = os.path.join(args["video_dir_path"], video_name)
        video = cv2.VideoCapture(video_path)

        frames = []
        for _, row in clip_df.iterrows():
            idx = row["pred_frame"]
            gt_label = row["class"]
            pred_label = labels[row["clip_name"]]
            pred_prob = probs[row["clip_name"]]

            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()

            if not success:
                break

            if gt_label == 1:
                cv2.putText(
                    frame,
                    f"true: {CLASSES[1]}",
                    (100, 100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2,
                    (0, 0, 255),
                    4,
                )
            else:
                cv2.putText(
                    frame,
                    f"true: {CLASSES[0]}",
                    (100, 100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2,
                    (0, 255, 0),
                    4,
                )

            if pred_label == 1:
                if pred_prob > args["pred_thr"]:
                    cv2.putText(
                        frame,
                        f"pred: {CLASSES[1]} ({pred_prob:.2f})",
                        (80, 160),
                        cv2.FONT_HERSHEY_DUPLEX,
                        2,
                        (0, 0, 255),
                        4,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"pred: {CLASSES[0]} ({pred_prob:.2f})",
                        (80, 160),
                        cv2.FONT_HERSHEY_DUPLEX,
                        2,
                        (0, 255, 0),
                        4,
                    )
            else:
                cv2.putText(
                    frame,
                    f"pred: {CLASSES[0]}",
                    (80, 160),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2,
                    (0, 255, 0),
                    4,
                )

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        video.release()

        # save_video_path = os.path.join(args["save_dir_path"], video_name)
        new_video = ImageSequenceClip(frames, fps=2)
        # new_video.write_videofile(
        #     save_video_path, codec="libx264", logger=None
        # )
        videos.append(new_video)

    concat_video = concatenate_videoclips(videos)
    concat_video.write_videofile(
        os.path.join(args["save_dir_path"], "concat_video.mp4"),
        codec="libx264",
        logger=None,
    )


def calc_metric(args, labels, probs):
    clips_df = pd.read_csv(args["clip_anno_path"])

    TP, FP, FN = 0, 0, 0
    for video_name in tqdm(
        os.listdir(args["video_dir_path"]), desc="[Calc event metric]"
    ):
        clip_df = clips_df[clips_df["video_name"] == video_name]
        clip_df = clip_df.sort_values(by="pred_frame")

        prev_class = 0
        predicted = False
        for _, row in clip_df.iterrows():
            clip_name = row["clip_name"]
            cur_class = row["class"]
            is_abnormal = (
                labels[clip_name] == 1 and probs[clip_name] > args["pred_thr"]
            )

            if cur_class == 1 and is_abnormal and not predicted:
                TP += 1
                predicted = True
            elif cur_class != 1:
                if prev_class == 1 and not predicted:
                    FN += 1
                if is_abnormal and not predicted:
                    FP += 1
                predicted = False

            prev_class = cur_class

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    pred_thr = args["pred_thr"]
    print(
        f"pred_thr: {pred_thr:.2f} | Event f1: {f1:.4f} | Event precision: {precision:.4f} | Event recall: {recall:.4f}"
    )


if __name__ == "__main__":
    args = {
        "model_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/save/T4_F12_S320/exp6/best_model.pth",
        "save_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/save/T4_F12_S320/exp6",
        "clip_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/clips/T4_F12_S320",
        "clip_anno_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/clip_anno_t4_f12_s320.csv",
        "video_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/videos",
        "pred_thr": 0.8,
    }

    if not os.path.exists(args["save_dir_path"]):
        os.makedirs(args["save_dir_path"])

    # define model
    model = CNNFCNN(frame=12)

    labels, probs = predict(args, model)
    visualize(args, labels, probs)
    calc_metric(args, labels, probs)

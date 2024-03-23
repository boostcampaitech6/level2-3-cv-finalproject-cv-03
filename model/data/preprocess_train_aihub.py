import os
import random
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm
import ast


TOTAL_FRAME = 180
FPS = 3


# labeling csv로 각 비디오의 프레임별 클래스 정보 생성후 어노테이션 파일로 저장
def make_video_annotation(video_dir_path, labeling_csv_path, video_anno_path):
    video_labels = {}
    for video_name in os.listdir(video_dir_path):
        video_labels[video_name] = [0 for _ in range(TOTAL_FRAME)]

    labeling_df = pd.read_csv(labeling_csv_path)
    for _, row in labeling_df.iterrows():
        video_name = row["file_list"][2:-2]
        start_frame_idx = int(row["temporal_segment_start"] * FPS)
        end_frame_idx = int(row["temporal_segment_end"] * FPS)

        video_labels[video_name][start_frame_idx : end_frame_idx + 1] = [1] * (
            end_frame_idx - start_frame_idx + 1
        )

    pd.DataFrame(
        list(video_labels.items()), columns=["video_name", "labels"]
    ).to_csv(video_anno_path, index=False)

    print(f"[DONE] Save video annotation file at {video_anno_path}")


# 클립 생성을 위해 각 비디오에서 같은 비율로 normal과 abnormal frame idx 추출
def sample_clip_idx(video_anno_path, clip_sec):
    anno_df = pd.read_csv(video_anno_path)
    clip_len = int(clip_sec * FPS)
    normal_clip_idx, abnormal_clip_idx = {}, {}

    for _, row in anno_df.iterrows():
        video_name = row["video_name"]
        labels = ast.literal_eval(row["labels"])

        normal_indices, abnormal_range = [], []
        prev_label, abnormal_st, abnormal_end = 0, 0, 0
        for idx, label in enumerate(labels):
            if idx < clip_len - 1:
                continue

            if label == 0:
                normal_indices.append(idx)
                if prev_label == 1:
                    abnormal_end = idx - 1
                    abnormal_range.append((abnormal_st, abnormal_end))
            elif label == 1 and prev_label == 0:
                abnormal_st = idx

            prev_label = label

        abnormal_indices = []
        for st, end in abnormal_range:
            sample_k = min(2, end - st + 1)
            abnormal_indices.extend(
                random.sample(range(st, end + 1), sample_k)
            )

        abnormal_clip_idx[video_name] = abnormal_indices
        normal_clip_idx[video_name] = random.sample(
            normal_indices, len(abnormal_indices)
        )

    print("[DONE] Sample clip index.")

    return normal_clip_idx, abnormal_clip_idx


# 선택된 인덱스로 실제 클립별 프레임을 추출하여 npy 파일로 저장하고 클립 어노테이션 생성
def save_clip_frames(
    video_dir_path,
    normal_clip_idx,
    abnormal_clip_idx,
    clip_dir_path,
    clip_anno_path,
    clip_sec=5,
    clip_frame=16,
    frame_size=(640, 640),
):
    clip_len = int(clip_sec * FPS)
    clip_annotation = {"video_name": [], "clip_name": [], "class": []}

    for video_name in tqdm(
        os.listdir(video_dir_path), desc="[Save clip frames]"
    ):
        video_path = os.path.join(video_dir_path, video_name)
        video = cv2.VideoCapture(video_path)

        for label, clip_idx in enumerate(
            [normal_clip_idx[video_name], abnormal_clip_idx[video_name]]
        ):
            for idx in clip_idx:
                start_idx = idx - clip_len + 1
                end_idx = idx

                indices = np.linspace(start_idx, end_idx, clip_frame)
                indices = np.round(indices).astype(int)

                frames = []
                for idx in indices:
                    video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    success, frame = video.read()

                    if not success:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = A.Resize(frame_size[0], frame_size[1])(
                        image=frame
                    )["image"]
                    frames.append(frame)

                clip_name = f"{os.path.splitext(video_name)[0]}_{idx}.npy"
                np.save(os.path.join(clip_dir_path, clip_name), frames)

                clip_annotation["video_name"].append(video_name)
                clip_annotation["clip_name"].append(clip_name)
                clip_annotation["class"].append(label)

        video.release()

    pd.DataFrame(clip_annotation).to_csv(clip_anno_path, index=False)


def main(args):
    # # video annotation file이 없을 때만 실행
    # make_video_annotation(
    #     args["video_dir_path"],
    #     args["labeling_csv_path"],
    #     args["video_anno_path"],
    # )

    normal_clip_idx, abnormal_clip_idx = sample_clip_idx(
        args["video_anno_path"], args["clip_sec"]
    )

    save_clip_frames(
        video_dir_path=args["video_dir_path"],
        normal_clip_idx=normal_clip_idx,
        abnormal_clip_idx=abnormal_clip_idx,
        clip_dir_path=args["clip_dir_path"],
        clip_anno_path=args["clip_anno_path"],
        clip_sec=args["clip_sec"],
        clip_frame=args["clip_frame"],
        frame_size=args["frame_size"],
    )


if __name__ == "__main__":
    root_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train"
    args = {
        "labeling_csv_path": os.path.join(
            root_dir, "labeling_aihub_train.csv"
        ),
        "video_dir_path": os.path.join(root_dir, "videos"),
        "clip_sec": 5,
        "clip_frame": 16,
        "frame_size": (640, 640),
    }

    args["video_anno_path"] = os.path.join(
        root_dir, "video_anno_aihub_train.csv"
    )

    args["clip_dir_path"] = os.path.join(
        root_dir,
        "clips",
        f"T{args['clip_sec']}_F{args['clip_frame']}_S{args['frame_size'][0]}",
    )
    args["clip_anno_path"] = os.path.join(
        root_dir,
        f"clip_anno_t{args['clip_sec']}_f{args['clip_frame']}_s{args['frame_size'][0]}.csv",
    )

    if not os.path.exists(args["clip_dir_path"]):
        os.makedirs(args["clip_dir_path"])

    main(args)

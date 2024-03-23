import os
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


# 1초 간격으로 이동하면서 프레임을 추출하여 npy 파일로 저장하고 클립 어노테이션 생성
def save_clip_frames(
    video_dir_path,
    video_anno_path,
    clip_dir_path,
    clip_anno_path,
    clip_sec=5,
    clip_frame=16,
    frame_size=(640, 640),
):
    video_anno_df = pd.read_csv(video_anno_path)
    clip_len = int(clip_sec * FPS)
    clip_annotation = {
        "video_name": [],
        "clip_name": [],
        "class": [],
        "pred_frame": [],
    }

    for _, row in tqdm(video_anno_df.iterrows(), desc="[Save clip frames]"):
        video_name = row["video_name"]
        video_path = os.path.join(video_dir_path, video_name)
        video = cv2.VideoCapture(video_path)
        labels = ast.literal_eval(row["labels"])

        for idx in range(clip_len - 1, TOTAL_FRAME, FPS):
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
                frame = A.Resize(frame_size[0], frame_size[1])(image=frame)[
                    "image"
                ]
                frames.append(frame)

            clip_name = (
                f"{os.path.splitext(video_name)[0]}_{(idx // FPS) + 1}.npy"
            )
            np.save(os.path.join(clip_dir_path, clip_name), frames)

            clip_annotation["video_name"].append(video_name)
            clip_annotation["clip_name"].append(clip_name)
            clip_annotation["class"].append(labels[idx])
            clip_annotation["pred_frame"].append(idx)

        video.release()

    pd.DataFrame(clip_annotation).to_csv(clip_anno_path, index=False)


def main(args):
    # # video annotation file이 없을 때만 실행
    # make_video_annotation(
    #     args["video_dir_path"],
    #     args["labeling_csv_path"],
    #     args["video_anno_path"],
    # )

    save_clip_frames(
        video_dir_path=args["video_dir_path"],
        video_anno_path=args["video_anno_path"],
        clip_dir_path=args["clip_dir_path"],
        clip_anno_path=args["clip_anno_path"],
        clip_sec=args["clip_sec"],
        clip_frame=args["clip_frame"],
        frame_size=args["frame_size"],
    )


if __name__ == "__main__":
    root_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid"
    args = {
        "labeling_csv_path": os.path.join(
            root_dir, "labeling_aihub_valid.csv"
        ),
        "video_dir_path": os.path.join(root_dir, "videos"),
        "clip_sec": 5,
        "clip_frame": 16,
        "frame_size": (640, 640),
    }

    args["video_anno_path"] = os.path.join(
        root_dir, "video_anno_aihub_valid.csv"
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

import os
import random
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm


def get_video_info(video_path):
    video_info = {}
    video_name = os.listdir(video_path)

    for video in video_name:
        video_read = cv2.VideoCapture(os.path.join(video_path, video))
        video_frame = int(video_read.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(video_read.get(cv2.CAP_PROP_FPS))

        video_info[video] = [video_frame, video_fps]
        video_read.release()

    return video_info


PATH = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/videos"
VIDEO_INFO = get_video_info(PATH)


# labeling csv로 각 비디오의 프레임별 클래스 정보 생성후 어노테이션 파일로 저장
def make_video_annotation(video_dir_path, labeling_csv_path, video_anno_path):
    video_labels = {}
    for video_name in os.listdir(video_dir_path):
        TOTAL_FRAME = VIDEO_INFO[video_name][0]
        FPS = VIDEO_INFO[video_name][1]
        
        video_labels[video_name] = [0 for _ in range(TOTAL_FRAME)]

    labeling_df = pd.read_csv(labeling_csv_path)
    abnormal_frame_idx, normal_frame_idx = {}, {}

    for _, row in labeling_df.iterrows():
        video_name = row["file_list"][2:-2]

        if abnormal_frame_idx.get(video_name) is None:
            abnormal_frame_idx[video_name] = []
        
        if normal_frame_idx.get(video_name) is None:
            normal_frame_idx[video_name] = []

        start_frame_idx = int(row["temporal_segment_start"] * FPS)
        end_frame_idx = int(row["temporal_segment_end"] * FPS)

        if row["metadata"] == '{"Class":"1"}':
            abnormal_frame_idx[video_name].append((start_frame_idx, end_frame_idx))
            video_labels[video_name][start_frame_idx : end_frame_idx + 1] = [1] * (end_frame_idx - start_frame_idx + 1)
        else:
            normal_frame_idx[video_name].append([i for i in range(start_frame_idx, end_frame_idx+1)])

    pd.DataFrame(
        list(video_labels.items()), columns=["video_name", "labels"]
    ).to_csv(video_anno_path, index=False)

    print(f"[DONE] Save video annotation file at {video_anno_path}")

    return abnormal_frame_idx, normal_frame_idx


# 클립 생성을 위해 각 비디오에서 같은 비율로 normal과 abnormal frame idx 추출
def sample_clip_idx(abnormal_frame_idx, normal_frame_idx, clip_sec):
    normal_clip_idx, abnormal_clip_idx = {}, {}

    for video_name in abnormal_frame_idx.keys():
        FPS = VIDEO_INFO[video_name][1]
        clip_len = int(clip_sec * FPS)

        abnormal_range = []
        abnormal_st, abnormal_end = 0, 0
        for st, end in abnormal_frame_idx[video_name]:
            abnormal_st = st
            for _ in range(st, end+1):
                if st + clip_len <= end:
                    abnormal_end = abnormal_st + clip_len
                    abnormal_range.append((abnormal_st, abnormal_end))
                    abnormal_st = abnormal_end + 1
                else:
                    continue
        
        abnormal_indices = []
        for st, end in abnormal_range:
            sample_k = min(2, end - st + 1)
            abnormal_indices.extend(
                random.sample(range(st, end + 1), sample_k)
            )
        
        abnormal_clip_idx[video_name] = abnormal_indices
        normal_clip_idx[video_name] = random.sample(
            normal_frame_idx, len(abnormal_indices)
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
    clip_annotation = {"video_name": [], "clip_name": [], "class": []}

    for video_name in tqdm(
        os.listdir(video_dir_path), desc="[Save clip frames]"
    ):
        FPS = VIDEO_INFO[video_name][1]
        clip_len = int(clip_sec * FPS)
        
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
    abnormal_frame_idx, normal_frame_idx = make_video_annotation(
        args["video_dir_path"],
        args["labeling_csv_path"],
        args["video_anno_path"],
    )

    normal_clip_idx, abnormal_clip_idx = sample_clip_idx(
        abnormal_frame_idx, normal_frame_idx, args["clip_sec"]
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
            root_dir, "ucf_labeling.csv"
        ),
        "video_dir_path": os.path.join(root_dir, "videos"),
        "clip_sec": 5,
        "clip_frame": 16,
        "frame_size": (640, 640),
    }

    args["video_anno_path"] = os.path.join(
        root_dir, "video_anno_ucf_train.csv"
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

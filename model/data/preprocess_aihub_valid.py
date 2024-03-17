import os
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm


TOTAL_FRAME = 180
TOTAL_SEC = 60
FPS = 3


# sec 정보를 frame 정보로 변환
def sec_to_frame(video_dir_path, labeling_csv_path, video_anno_path):
    video_labels = {}
    for video_name in os.listdir(video_dir_path):
        video_labels[video_name] = [0 for _ in range(TOTAL_FRAME)]

    labeling_df = pd.read_csv(labeling_csv_path)

    for _, row in tqdm(
        labeling_df.iterrows(),
        total=len(labeling_df),
        desc="[Convert sec to frame]",
    ):
        video_name = row["file_list"][2:-2]

        start_frame_idx = int(row["temporal_segment_start"] * FPS)
        end_frame_idx = int(row["temporal_segment_end"] * FPS)

        video_labels[video_name][start_frame_idx : end_frame_idx + 1] = [1] * (
            end_frame_idx - start_frame_idx + 1
        )

    pd.DataFrame(
        list(video_labels.items()), columns=["video_name", "labels"]
    ).to_csv(video_anno_path, index=False)

    return video_labels


# 1초 단위로 이동하면서 프레임을 추출하고 npy 파일과 annotation 파일 저장
def sample_frames(
    video_dir_path,
    clip_dir_path,
    clip_anno_path,
    video_labels,
    clip_sec=3,
    clip_frame=16,
    frame_size=(640, 640),
):
    clip_len = int(clip_sec * FPS)
    clip_annotation = {
        "video_name": [],
        "clip_name": [],
        "class": [],
        "pred_frame": [],
    }

    for i, video_name in enumerate(os.listdir(video_dir_path), start=1):
        video_path = os.path.join(video_dir_path, video_name)
        video = cv2.VideoCapture(video_path)

        for idx in tqdm(
            range(0, TOTAL_FRAME - clip_len + 1, FPS),
            desc=f"[Sample frames {i}/{len(os.listdir(video_dir_path))}]",
        ):
            start_idx = idx
            end_idx = idx + clip_len - 1

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

            frames = np.transpose(np.array(frames), (0, 3, 1, 2))

            clip_name = (
                f"{os.path.splitext(video_name)[0]}_{(end_idx + 1) // FPS}.npy"
            )
            np.save(os.path.join(clip_dir_path, clip_name), frames)

            labels = video_labels[video_name]
            sample_labels = labels[start_idx : end_idx + 1]

            if sample_labels[-1] == 1:
                clip_class = 1
            elif any(label == 1 for label in sample_labels):
                clip_class = 2
            else:
                clip_class = 0

            clip_annotation["video_name"].append(video_name)
            clip_annotation["clip_name"].append(clip_name)
            clip_annotation["class"].append(clip_class)
            clip_annotation["pred_frame"].append(end_idx)

        video.release()

    pd.DataFrame(clip_annotation).to_csv(clip_anno_path, index=False)


def main(args):
    video_labels = sec_to_frame(
        args["video_dir_path"],
        args["labeling_csv_path"],
        args["video_anno_path"],
    )
    sample_frames(
        video_dir_path=args["video_dir_path"],
        clip_dir_path=args["clip_dir_path"],
        clip_anno_path=args["clip_anno_path"],
        video_labels=video_labels,
        clip_sec=args["clip_sec"],
        clip_frame=args["clip_frame"],
        frame_size=args["frame_size"],
    )


if __name__ == "__main__":
    root_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid"
    args = {
        "labeling_csv_path": os.path.join(root_dir, "aihub_labeling_DY.csv"),
        "video_dir_path": os.path.join(root_dir, "videos"),
        "clip_sec": 4,
        "clip_frame": 12,
        "frame_size": (640, 640),
    }

    args["clip_dir_path"] = os.path.join(
        root_dir,
        "clips",
        f"T{args['clip_sec']}_F{args['clip_frame']}_S{args['frame_size'][0]}",
    )
    args["clip_anno_path"] = os.path.join(
        root_dir,
        f"clip_anno_t{args['clip_sec']}_f{args['clip_frame']}_s{args['frame_size'][0]}.csv",
    )
    args["video_anno_path"] = os.path.join(root_dir, "video_anno_aihub_DY.csv")

    if not os.path.exists(args["clip_dir_path"]):
        os.makedirs(args["clip_dir_path"])

    main(args)

import os
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm
import ast


def main(params, paths):
    anno_df = pd.read_csv(paths["anno_path"])
    anno_clip = {
        "video_name": [],
        "clip_name": [],
        "class": [],
        "pred_t": [],
    }

    for _, row in tqdm(
        anno_df.iterrows(),
        total=len(anno_df),
        desc="[Create anno_clip_val.csv]",
    ):
        video_name = row["video_name"]
        label = ast.literal_eval(row["label"])

        video_path = os.path.join(paths["video_dir_path"], video_name)
        video = cv2.VideoCapture(video_path)

        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_len = int(total_frame / fps)

        for t in range(total_len - params["clip_len"] + 1):
            indices = np.linspace(
                t * fps,
                (t + params["clip_len"]) * fps - 1,
                params["clip_frame"],
            )
            indices = np.round(indices).astype(int)

            frames = []
            for idx in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = video.read()

                if not success:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = A.Resize(params["frame_size"], params["frame_size"])(
                    image=frame
                )["image"]
                frames.append(frame)

            frames = np.transpose(np.array(frames), (0, 3, 1, 2))

            clip_name = f"{os.path.splitext(video_name)[0]}_{t}.npy"
            np.save(
                os.path.join(paths["save_dir_path"], clip_name),
                frames,
            )

            sample_labels = label[t : t + params["clip_len"]]

            if sample_labels[-1] == 1:
                clip_class = 1
            elif any(label == 1 for label in sample_labels):
                clip_class = 2
            elif any(label == 0 for label in sample_labels):
                clip_class = 0
            else:
                clip_class = 3

            anno_clip["video_name"].append(video_name)
            anno_clip["video_name"].append(clip_name)
            anno_clip["class"].append(clip_class)
            anno_clip["pred_t"].append(t + params["clip_len"])

        video.release()

    pd.DataFrame(anno_clip).to_csv(paths["save_anno_path"], index=False)


if __name__ == "__main__":
    params = {
        "clip_len": 5,
        "clip_frame": 16,
        "frame_size": 224,
    }
    paths = {
        "anno_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/anno_video_val.csv",
        "video_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/videos",
        "save_dir_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/clips/T{params['clip_len']}_F{params['clip_frame']}_S{params['frame_size']}",
        "save_anno_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/anno_clip_val_t{params['clip_len']}_f{params['clip_frame']}_s{params['frame_size']}.csv",
    }

    if not os.path.exists(paths["save_dir_path"]):
        os.makedirs(paths["save_dir_path"])

    main(params, paths)

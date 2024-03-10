import os
import random
import numpy as np
import pandas as pd
import cv2
import albumentations as A


def main(params, paths):
    anno_df = pd.read_csv(paths["anno_path"])
    meta_df = pd.read_csv(paths["meta_path"])
    new_anno_dict = {"file_name": [], "class": []}

    for _, row in meta_df.iterrows():
        file_name = row["file_name"]
        start_t = row["sample_start_t"]
        end_t = row["sample_end_t"]
        sample_t = [
            random.sample(range(start_t, end_t + 1))
            for _ in range(params["sample_num"])
        ]

        video_path = os.path.join(paths["video_dir_path"], file_name)
        video = cv2.VideoCapture(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        for t in sample_t:
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
            np.save(
                os.path.join(
                    paths["save_dir_path"],
                    f"{os.path.splitext(file_name)[0]}_{t}.npy",
                ),
                frames,
            )

            labels = anno_df.loc[anno_df["file_name"] == file_name, "label"]
            sample_labels = labels[t : t + params["clip_len"]]

            if sample_labels[-1] == 1:
                clip_class = 1
            elif np.any(sample_labels == 1):
                clip_class = 2
            elif np.any(sample_labels == 0):
                clip_class = 0
            else:
                clip_class = 3

            new_anno_dict["file_name"].append(
                f"{os.path.splitext(file_name)[0]}_{t}.npy"
            )
            new_anno_dict["class"].append(clip_class)

        video.release()

    new_anno_df = pd.DataFrame(new_anno_dict)
    new_anno_df.to_csv(paths["new_anno_path"], index=False)


if __name__ == "__main__":
    params = {
        "clip_len": 5,
        "clip_frame": 16,
        "frame_size": 224,
        "sample_num": 2,
    }
    paths = {
        "meta_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/metadata_t{params['clip_len']}.csv",
        "anno_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/annotation.csv",
        "video_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/videos",
        "save_dir_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/frames/T{params['clip_len']}_F{params['clip_frame']}_S{params['frame_size']}",
        "new_anno_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/annotation_t{params['clip_len']}_f{params['clip_frame']}_s{params['frame_size']}.csv",
    }

    if not os.path.exists(paths["save_dir_path"]):
        os.makedirs(paths["save_dir_path"])

    main(params, paths)

import os
import numpy as np
import pandas as pd
import cv2
import albumentations as A
import ast


def main(params, paths):
    anno_df = pd.read_csv(paths["anno_path"])
    new_anno_dict = {"file_name": [], "class": [], "pred_t": []}

    for _, row in anno_df.iterrows():
        file_name = row["file_name"]
        label = ast.literal_eval(row["label"])

        video_path = os.path.join(paths["video_dir_path"], file_name)
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
            np.save(
                os.path.join(
                    paths["save_dir_path"],
                    f"{os.path.splitext(file_name)[0]}_{t}.npy",
                ),
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

            new_anno_dict["file_name"].append(
                f"{os.path.splitext(file_name)[0]}_{t}.npy"
            )
            new_anno_dict["class"].append(clip_class)
            new_anno_dict["pred_t"].append(t + params["clip_len"])

        video.release()

    new_anno_df = pd.DataFrame(new_anno_dict)
    new_anno_df.to_csv(paths["new_anno_path"], index=False)


if __name__ == "__main__":
    params = {
        "clip_len": 5,
        "clip_frame": 16,
        "frame_size": 224,
    }
    paths = {
        "anno_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/annotation.csv",
        "video_dir_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/videos",
        "save_dir_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/frames/T{params['clip_len']}_F{params['clip_frame']}_S{params['frame_size']}",
        "new_anno_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/annotation_t{params['clip_len']}_f{params['clip_frame']}_s{params['frame_size']}.csv",
    }

    if not os.path.exists(paths["save_dir_path"]):
        os.makedirs(paths["save_dir_path"])

    main(params, paths)

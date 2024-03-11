import json
import pandas as pd
from tqdm import tqdm


def main(path, clip_len):
    labeling_df = pd.read_csv(path["labeling_path"])
    video_names = [i.strip('[]"') for i in labeling_df["file_list"]]
    video_names_set = list(set(labeling_df["file_list"]))

    start_t = labeling_df["temporal_segment_start"]
    end_t = labeling_df["temporal_segment_end"]

    labels = {}
    dataset, class_id, sample_start_t, sample_end_t = [], [], [], []
    class_info = [json.loads(i) for i in labeling_df["metadata"]]

    # metadata
    for info, st, et in tqdm(
        zip(class_info, start_t, end_t),
        total=len(class_info),
        desc=f"[Create metadata_t{clip_len}.csv]",
    ):
        if info.get("Normal") == "0":
            dataset.append("Normal")
            class_id.append(0)
            sample_start_t.append(st)
            sample_end_t.append(et - clip_len)

        elif info.get("Abnormal") == "0":
            dataset.append("Abnormal")
            class_id.append(1)
            sample_start_t.append(max(st - clip_len + 1, 0))
            sample_end_t.append(min(et - 1, 60 - clip_len))

        elif info.get("Abnormal") == "1":
            dataset.append("Abnormal")
            class_id.append(3)
            sample_start_t.append(-1)
            sample_end_t.append(-1)

    # annotation
    for video_name in tqdm(video_names_set, desc="[Create anno_video.csv]"):
        file_df = labeling_df.loc[labeling_df["file_list"] == video_name]
        video_name = video_name.strip('[]"')

        labels[video_name] = [0 for _ in range(60)]

        for _, row in file_df.iterrows():
            if row["metadata"] == '{"Normal":"0"}':
                break
            elif row["metadata"] == '{"Abnormal":"0"}':
                for i in range(
                    row["temporal_segment_start"],
                    row["temporal_segment_end"],
                ):
                    labels[video_name][i] = 1
            elif row["metadata"] == '{"Abnormal":"1"}':
                for i in range(
                    row["temporal_segment_start"],
                    row["temporal_segment_end"],
                ):
                    labels[video_name][i] = 3

    # save csv
    meta_df = pd.DataFrame(
        {
            "dataset": dataset,
            "video_name": video_names,
            "class": class_id,
            "start_t": start_t,
            "end_t": end_t,
            "sample_start_t": sample_start_t,
            "sample_end_t": sample_end_t,
        }
    )

    anno_df = pd.DataFrame(
        {
            "video_name": list(labels.keys()),
            "labels": list(labels.values()),
        }
    )

    meta_df.to_csv(path["save_meta_path"], index=False)
    anno_df.to_csv(path["save_anno_path"], index=False)


if __name__ == "__main__":
    clip_len = 5

    path = {
        "labeling_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/aihub_labeling.csv",
        "save_meta_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/metadata_t{clip_len}.csv",
        "save_anno_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/anno_video.csv",
    }

    main(path, clip_len)

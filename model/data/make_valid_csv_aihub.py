import pandas as pd
from tqdm import tqdm


def main(labeling_path, save_anno_path):
    labeling_df = pd.read_csv(labeling_path)
    video_names = list(set(labeling_df["file_list"]))
    labels = {}

    for video_name in tqdm(video_names, desc="[Create anno_video_val.csv]"):
        file_df = labeling_df.loc[labeling_df["file_list"] == video_name]
        video_name = video_name.strip('[]"')

        labels[video_name] = [0 for _ in range(60)]

        for _, row in file_df.iterrows():
            if row["metadata"] == '{"Abnormal":"0"}':
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

    anno_df = pd.DataFrame(
        {
            "video_name": list(labels.keys()),
            "labels": list(labels.values()),
        }
    )
    anno_df.to_csv(save_anno_path, index=False)


if __name__ == "__main__":
    labeling_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/aihub_labeling_val.csv"
    save_anno_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/anno_video_val.csv"

    main(labeling_path, save_anno_path)

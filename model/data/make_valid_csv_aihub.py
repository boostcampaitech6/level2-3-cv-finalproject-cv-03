import pandas as pd


def main(labeling_csv_path, new_anno_path):
    labeling_df = pd.read_csv(labeling_csv_path)
    file_names = list(set(labeling_df["file_list"]))
    labels = {}

    for file_name in file_names:
        file_df = labeling_df.loc[labeling_df["file_list"] == file_name]
        file_name = file_name.strip('[]"')

        labels[file_name] = [0 for _ in range(60)]

        for _, row in file_df.iterrows():
            if row["metadata"] == '{"Abnormal":"0"}':
                for i in range(
                    row["temporal_segment_start"],
                    row["temporal_segment_end"],
                ):
                    labels[file_name][i] = 1
            elif row["metadata"] == '{"Abnormal":"1"}':
                for i in range(
                    row["temporal_segment_start"],
                    row["temporal_segment_end"],
                ):
                    labels[file_name][i] = 3

    new_anno_df = pd.DataFrame(
        {
            "file_name": list(labels.keys()),
            "label": list(labels.values()),
        }
    )
    new_anno_df.to_csv(new_anno_path, index=False)


if __name__ == "__main__":
    labeling_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/aihub_labeling_valid.csv"
    new_anno_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/annotation.csv"

    main(labeling_path, new_anno_path)

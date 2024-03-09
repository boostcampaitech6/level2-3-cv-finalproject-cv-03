import json
import pandas as pd


def main(path, clip_len):
    csv_data = pd.read_csv(path["labeling_csv_path"])
    file_name = [i.strip('[]"') for i in csv_data["file_list"]]
    remove_file_name = list(set(csv_data["file_list"]))

    start_t = csv_data["temporal_segment_start"]
    end_t = csv_data["temporal_segment_end"]

    label = {}
    dataset, class_idx, sample_start_t, sample_end_t = [], [], [], []
    class_info = [json.loads(i) for i in csv_data["metadata"]]

    # metadata
    for info, st, et in zip(class_info, start_t, end_t):
        if info["Normal"] == 0:
            dataset.append("Normal")
            class_idx.append(0)
            sample_start_t.append(st)
            sample_end_t.append(et - clip_len)

        elif info["Abnormal"] == "0":
            dataset.append("Abnormal")
            class_idx.append(1)
            sample_start_t.append(max(st - clip_len + 1, 0))
            sample_end_t.append(min(et - 1, 60 - clip_len))

        elif info["Abnormal"] == "1":
            dataset.append("Abnormal")
            class_idx.append(3)
            sample_start_t.append(-1)
            sample_end_t.append(-1)

    # annotation
    for name in remove_file_name:
        file_data = csv_data[csv_data["file_list"] == name]
        new_name = name.strip('[]"')

        label[new_name] = [0 for _ in range(60)]

        for _, row in file_data.iterrows():
            if row["metadata"] == '{"Normal":"0"}':
                break
            elif row["metadata"] == '{"Abnormal":"0"}':
                for i in range(
                    row["temporal_segment_start"],
                    row["temporal_segment_end"],
                ):
                    label[new_name][i] = 1
            elif row["metadata"] == '{"Abnormal":"1"}':
                for i in range(
                    row["temporal_segment_start"],
                    row["temporal_segment_end"],
                ):
                    label[new_name][i] = 3

    # save csv
    meta_df = pd.DataFrame(
        {
            "dataset": dataset,
            "file_name": file_name,
            "class": class_idx,
            "start_t": start_t,
            "end_t": end_t,
            "sample_start_t": sample_start_t,
            "sample_end_t": sample_end_t,
        }
    )

    anno_df = pd.DataFrame(
        {
            "file_name": list(label.keys()),
            "label": list(label.values()),
        }
    )

    meta_df.to_csv(path["meta_csv_path"], index=False)
    anno_df.to_csv(path["anno_csv_path"], index=False)


if __name__ == "__main__":
    clip_len = 5

    path = {
        "labeling_csv_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/anno_30.csv",
        "meta_csv_path": f"/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/metadata_t{clip_len}.csv",
        "anno_csv_path": "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/annotation.csv",
    }

    main(path, clip_len)

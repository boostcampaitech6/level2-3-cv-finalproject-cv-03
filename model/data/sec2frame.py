import os
import cv2
import pandas as pd


def main(video_path, anno_csv_path, save_csv_path):
    csv_data = pd.read_csv(anno_csv_path)
    file_names = [i.strip('[]"') for i in csv_data["file_list"]]
    start_list = csv_data["temporal_segment_start"]
    end_list = csv_data["temporal_segment_end"]

    # second
    start_sec = [(int(round(s))) for s in start_list]
    end_sec = [(int(round(e))) for e in end_list]

    # fps
    fps = {}
    for video_name in os.listdir(video_path):
        video = os.path.join(video_path, video_name)
        video = cv2.VideoCapture(video)
        fps[video_name] = video.get(cv2.CAP_PROP_FPS)
        video.release()

    # frame
    start_frame = [
        (int(round(start) * fps[name]))
        for name, start in zip(file_names, start_list)
    ]
    end_frame = [
        (int(round(end) * fps[name]))
        for name, end in zip(file_names, end_list)
    ]

    # class
    class_names = []
    for data in csv_data["metadata"]:
        if data == '{"Shoplifting":"0"}':
            class_names.append("Normal")
        elif data == '{"Shoplifting":"1"}':
            class_names.append("Shoplifting")
        elif data == '{"Shoplifting":"2"}':
            class_names.append("Background")

    # csv
    df = pd.DataFrame(
        {
            "file_name": video,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "class": class_names,
        }
    )

    df.to_csv(save_csv_path, index=False)


if __name__ == "__main__":
    video_path = (
        "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/dataset/videos"
    )
    anno_csv_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/dataset/annotation.csv"
    save_csv_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/dataset/anno_frame.csv"

    main(video_path, anno_csv_path, save_csv_path)

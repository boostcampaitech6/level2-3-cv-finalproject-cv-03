import os
import cv2
import pandas as pd


def extractor(video_path, csv_path):
    csv_data = pd.read_csv(csv_path)
    file_names = [i.strip('[]"') for i in csv_data["file_list"]]

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
        for name, start in zip(file_names, csv_data["temporal_segment_start"])
    ]
    end_frame = [
        (int(round(end) * fps[name]))
        for name, end in zip(file_names, csv_data["temporal_segment_end"])
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

    return file_names, start_frame, end_frame, class_names


video_path = (
    "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/dataset/videos"
)
csv_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/dataset/annotation.csv"
save_csv = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/dataset/anno_frame.csv"

video, start, end, class_ = extractor(video_path, csv_path)

df = pd.DataFrame(
    {
        "file_name": video,
        "start_frame": start,
        "end_frame": end,
        "class": class_,
    }
)

df.to_csv(save_csv, index=False)

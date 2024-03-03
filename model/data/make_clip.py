import os
import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip


def make_shoplifting_clip(anno_path, video_path, clip_path, metadata):
    anno_df = pd.read_csv(anno_path)
    fname2sec = {}
    for _, row in anno_df.iterrows():
        if row["class"] == "Shoplifting":
            file_name = row["file_name"]
            start_sec = row["start_sec"]
            end_sec = row["end_sec"]

            if fname2sec.get(file_name) is None:
                fname2sec[file_name] = [(start_sec, end_sec)]
            else:
                fname2sec[file_name].append((start_sec, end_sec))

    idx = 1
    for video_name in tqdm(os.listdir(video_path), desc="Shoplifting"):
        video = VideoFileClip(os.path.join(video_path, video_name))
        for start, end in fname2sec[video_name]:
            clip = video.subclip(start, end)

            clip_name = f"Shoplifting_{idx}.mp4"
            metadata["origin_fname"].append(video_name)
            metadata["clip_fname"].append(clip_name)
            metadata["sec"].append(clip.duration)

            clip.write_videofile(
                os.path.join(clip_path, clip_name), fps=30, logger=None
            )

            idx += 1

    meta_df = pd.DataFrame(metadata)
    clip_num = meta_df["sec"].count()
    median_sec = meta_df["sec"].median()

    return clip_num, median_sec


def make_normal_clip(anno_path, video_path, clip_path, metadata, interval):
    anno_df = pd.read_csv(anno_path)
    fname2sec = {}
    for _, row in anno_df.iterrows():
        if row["class"] == "Normal":
            file_name = row["file_name"]
            start_sec = row["start_sec"]
            end_sec = row["end_sec"]

            if fname2sec.get(file_name) is None:
                fname2sec[file_name] = [(start_sec, end_sec)]
            else:
                fname2sec[file_name].append((start_sec, end_sec))

    idx = 1
    for video_name in tqdm(os.listdir(video_path), desc="Normal"):
        clip_sec = []
        for start, end in fname2sec[video_name]:
            cur = start
            while cur + interval <= end:
                clip_sec.append((cur, cur + interval))
                cur = cur + interval

        video = VideoFileClip(os.path.join(video_path, video_name))
        for start, end in clip_sec:
            clip = video.subclip(start, end)

            clip_name = f"Normal_{idx}.mp4"
            metadata["origin_fname"].append(video_name)
            metadata["clip_fname"].append(clip_name)
            metadata["sec"].append(clip.duration)

            clip.write_videofile(
                os.path.join(clip_path, clip_name), fps=30, logger=None
            )

            idx += 1


def main(anno_path, video_path, clip_path, meta_path):
    metadata = {"origin_fname": [], "clip_fname": [], "sec": []}

    shoplifting_clip_path = os.path.join(clip_path, "shoplifting")
    normal_clip_path = os.path.join(clip_path, "normal")

    if not os.path.exists(shoplifting_clip_path):
        os.makedirs(shoplifting_clip_path)
    if not os.path.exists(normal_clip_path):
        os.makedirs(normal_clip_path)

    clip_num, median_sec = make_shoplifting_clip(
        anno_path, video_path, shoplifting_clip_path, metadata
    )
    print(
        f"Shoplifting clip num: {clip_num} | Shoplifting median sec: {median_sec}"
    )

    make_normal_clip(
        anno_path, video_path, normal_clip_path, metadata, median_sec
    )

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(meta_path, index=False)


if __name__ == "__main__":
    anno_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/annotation.csv"
    video_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/videos"
    clip_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/clips"
    meta_path = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/metadata.csv"

    if not os.path.exists(clip_path):
        os.makedirs(clip_path)

    main(anno_path, video_path, clip_path, meta_path)

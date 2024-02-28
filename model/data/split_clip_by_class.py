import os
import pandas as pd
from moviepy.editor import VideoFileClip, concatenate_videoclips
from multiprocessing import Pool
from functools import partial


def split_clip(anno_df, fname2idx, video_path, clip_path, video_name):
    clips = {"Normal": [], "Shoplifting": [], "Background": []}

    for i in fname2idx[video_name]:
        start_sec = anno_df.loc[i]["start_sec"]
        end_sec = anno_df.loc[i]["end_sec"]
        clip_class = anno_df.loc[i]["class"]

        video = VideoFileClip(os.path.join(video_path, video_name))
        clips[clip_class].append(video.subclip(start_sec, end_sec))

    for key, value in clips.items():
        if value:
            concat_clip = concatenate_videoclips(value)
            concat_clip.write_videofile(
                os.path.join(clip_path[key], video_name)
            )
            # concat_clip.write_videofile(os.path.join(clip_path[key], video_name), fps=16)


def main(anno_csv_path, video_path, clip_path):
    anno_df = pd.read_csv(anno_csv_path)
    fname2idx = {}
    for i in range(anno_df.__len__()):
        fname = anno_df.loc[i]["file_name"]
        if fname2idx.get(fname) is None:
            fname2idx[fname] = [i]
        else:
            fname2idx[fname].append(i)

    partial_split_clip = partial(
        split_clip, anno_df, fname2idx, video_path, clip_path
    )

    with Pool() as pool:
        pool.map(partial_split_clip, os.listdir(video_path))


if __name__ == "__main__":
    anno_csv_path = "../dataset/anno_frame.csv"
    video_path = "../dataset/videos"
    clip_path = {
        "Normal": "../dataset/clips/normal",
        "Shoplifting": "../dataset/clips/shoplifting",
        "Background": "../dataset/clips/background",
    }

    for path in clip_path.values():
        if not os.path.exists(path):
            os.makedirs(path)

    main(anno_csv_path, video_path, clip_path)

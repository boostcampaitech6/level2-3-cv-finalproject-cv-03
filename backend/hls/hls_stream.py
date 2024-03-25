import os
import json
import shutil
import redis
import subprocess


def run_ffmpeg(cctv_config, hls_root_dir):
    global process_dict

    cctv_id, rtsp_url = str(cctv_config["cctv_id"]), cctv_config["cctv_url"]
    hls_stream_dir = os.path.join(hls_root_dir, cctv_id)
    os.makedirs(hls_stream_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-fflags",
        "nobuffer",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-vsync",
        "0",
        "-copyts",
        "-vcodec",
        "copy",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-an",
        "-hls_flags",
        "delete_segments+append_list",
        "-f",
        "hls",
        "-hls_time",
        "2",
        "-hls_list_size",
        "3",
        "-hls_segment_type",
        "mpegts",
        "-hls_segment_filename",
        f"{hls_stream_dir}/%3d.ts",
        f"{hls_stream_dir}/index.m3u8",
    ]

    process = subprocess.Popen(command)
    process_dict[cctv_id] = process


def stop_ffmpeg(cctv_id, hls_root_dir):
    global process_dict
    process_dict[cctv_id].kill()

    hls_stream_dir = os.path.join(hls_root_dir, str(cctv_id))
    shutil.rmtree(hls_stream_dir)


if __name__ == "__main__":
    global process_dict
    process_dict = dict()

    redis_server = redis.Redis(host="10.28.224.201", port=30435, db=0)
    hls_root_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/backend/hls/cctv_stream"

    start_hls_stream_key = "start_hls_stream"
    stop_hls_stream_key = "stop_hls_stream"
    while True:
        k, v = redis_server.brpop(
            keys=[start_hls_stream_key, stop_hls_stream_key], timeout=None
        )
        k, v = k.decode("utf-8"), v.decode("utf-8")
        if k == start_hls_stream_key:
            config = json.loads(v)
            print(f"hls streaming start.. : {config}")
            run_ffmpeg(config, hls_root_dir)
        elif k == stop_hls_stream_key:
            cctv_id = v
            print(f"hls streaming stop.. : {cctv_id}")
            stop_ffmpeg(cctv_id, hls_root_dir)

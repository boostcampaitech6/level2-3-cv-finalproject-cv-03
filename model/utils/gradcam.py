import os
import sys
import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from arch import *
from data import GradCAMDataset
from utils import RNN_INPUT_SIZE


def grad_cam(model, frame, target_layer):
    def forward_hook(module, input, output):
        grad_cam_data["feature_map"] = output

    def backward_hook(module, grad_input, grad_output):
        grad_cam_data["grad_output"] = grad_output[0]

    grad_cam_data = {}
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(frame)
    model.zero_grad()
    output[0, 1].backward()

    feature_map = grad_cam_data["feature_map"]
    grad_output = grad_cam_data["grad_output"]

    weights = grad_output.mean(dim=(2, 3), keepdim=True)

    cam = (weights * feature_map).sum(dim=1, keepdim=True).squeeze()
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.detach().cpu().numpy()

    return cam


def save_grad_cam_video(frames, heatmaps, save_video_path, frame_size):
    overlay_frames = []

    for frame, heatmap in tqdm(
        zip(frames, heatmaps), total=len(frames), desc="[Save grad-cam video]"
    ):
        heatmap_resized = cv2.resize(heatmap, (frame_size, frame_size))
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(
            heatmap_normalized, cv2.COLORMAP_JET
        )

        overlay_frame = cv2.addWeighted(
            frame.astype("uint8"), 0.6, heatmap_colored, 0.4, 0
        )
        overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        overlay_frames.append(overlay_frame)

    save_video = ImageSequenceClip(overlay_frames, fps=1)
    save_video.write_videofile(save_video_path, codec="libx264", logger=None)


def get_frames(video_path, frame_size):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_cnt = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for idx in range(fps - 1, frame_cnt, fps):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = A.Resize(frame_size, frame_size)(image=frame)["image"]
        frames.append(frame)

    return frames


def main(args):
    frames = get_frames(args["video_path"], args["frame_size"])

    dataset = GradCAMDataset(frames)
    dataloaer = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = args["model"].to(device)

    heatmaps = []
    for frame in tqdm(dataloaer, desc="[Extract feature maps]"):
        frame = frame.to(device)
        heatmap = grad_cam(
            model, frame=frame, target_layer=args["target_layer"]
        )
        heatmaps.append(heatmap)

    save_video_path = os.path.join(
        args["save_dir_path"], os.path.basename(args["video_path"])
    )
    save_grad_cam_video(frames, heatmaps, save_video_path, args["frame_size"])


if __name__ == "__main__":
    args = {
        "model_path": "best_model.pth",
        "video_path": "video.mp4",
        "save_dir_path": "save/gradcam",
        "frame_size": 640,
    }

    if not os.path.exists(args["save_dir_path"]):
        os.makedirs(args["save_dir_path"])

    cnn = MobileNet()
    rnn = GRU(
        input_size=RNN_INPUT_SIZE[str(cnn)], hidden_size=512, num_layers=1
    )
    model = ClipModel(cnn, rnn, hidden=512, nc=3)
    model.load_state_dict(torch.load(args["model_path"]))

    target_layer = cnn.mobilenet[-1][0]

    args.update({"model": model, "target_layer": target_layer})

    main(args)

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import albumentations as A
import wandb

from arch import *
from data import *
from trainer import *
from utils import *


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(0)

    transform = A.Normalize()

    train_dataset = ClipTrainDataset(
        clip_dir_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/clips/T4_F12_S640",
        clip_anno_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/clip_anno_t4_f12_s640.csv",
        transform=transform,
    )
    valid_dataset = ClipTrainDataset(
        clip_dir_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/clips/T4_F12_S640",
        clip_anno_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/clip_anno_t4_f12_s640.csv",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=8
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=4, shuffle=False, num_workers=8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_name = "T4_F12_S640_exp1"
    save_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/save/T4_F12_S640/exp17"
    best_model_path = os.path.join(save_dir, "best_model.pth")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = CNNRNN()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0]).to(device))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    epochs = 100
    # best_loss = 1e9
    best_abnormal_f1 = 0
    early_stop = 5
    not_improved = 0

    train_metric_tracker = MetricTracker()
    valid_metric_tracker = MetricTracker()

    wandb.init(project="WatchDUCK!!", entity="rudeuns", name=wandb_name)

    for epoch in range(1, epochs + 1):
        print(f"\n---EPOCH {epoch}---")
        print("lr:", optimizer.param_groups[0]["lr"])

        train_scores = train_loop(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            train_metric_tracker,
            device,
            epoch,
        )
        valid_scores = valid_loop(
            model, valid_loader, criterion, valid_metric_tracker, device, epoch
        )

        log_data = {
            "train_loss": train_scores["loss"],
            "valid_loss": valid_scores["loss"],
        }

        for i, class_name in enumerate(["Normal", "Doubt"]):
            log_data.update(
                {
                    f"train_{class_name}_f1_score": train_scores["f1"][i],
                    f"train_{class_name}_precision": train_scores["precision"][
                        i
                    ],
                    f"train_{class_name}_recall": train_scores["recall"][i],
                    f"valid_{class_name}_f1_score": valid_scores["f1"][i],
                    f"valid_{class_name}_precision": valid_scores["precision"][
                        i
                    ],
                    f"valid_{class_name}_recall": valid_scores["recall"][i],
                }
            )

        wandb.log(log_data, step=epoch)

        print(
            f"Train | f1 score: {train_scores['f1']} | precision: {train_scores['precision']} | recall: {train_scores['recall']} | loss: {train_scores['loss']:.5f} |\n"
            f"Valid | f1 score: {valid_scores['f1']} | precision: {valid_scores['precision']} | recall: {valid_scores['recall']} | loss: {valid_scores['loss']:.5f} |"
        )

        # best model 저장
        if valid_scores["f1"][1] > best_abnormal_f1:
            not_improved = 0
            best_abnormal_f1 = valid_scores["f1"][1]
            torch.save(model.state_dict(), best_model_path)
            print(f"Save best model epoch {epoch}")
        else:
            not_improved += 1

        if not_improved == early_stop:
            print(f"Early stopped. Epoch {epoch}")
            break


if __name__ == "__main__":
    main()

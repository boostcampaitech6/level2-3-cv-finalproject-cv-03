import os
import tempfile
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from arch import YOLO, GRU, ClipModel, rnn_input_sizes
from data import ClipTrainDataset
from trainer import MetricTracker, train_loop, valid_loop

import wandb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():
    set_seed(seed=2024)

    train_dataset = ClipTrainDataset(
        frame_dir_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/frames/folder_name",
        anno_csv_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/train/anno_name.csv",
    )
    valid_dataset = ClipTrainDataset(
        frame_dir_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/frames/folder_name",
        anno_csv_path="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/valid/anno_name.csv",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=8
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = YOLO(
        cfg="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/arch/yolo_yaml/yolov8n-cls.yaml"
    )
    input_size = rnn_input_sizes[str(cnn)]
    rnn = GRU(input_size=input_size, hidden_size=512, num_layers=1)

    model = ClipModel(cnn, rnn)
    save_dir = "/data/ephemeral/home/level2-3-cv-finalproject-cv-03/model/dataset/save_dir"

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    best_loss = 1e9

    metric_tracker = MetricTracker()

    wandb.init(project="WatchDUCK!!", entity="gusdn00751")

    for epoch in range(epochs):
        train_scores = train_loop(
            model,
            train_loader,
            optimizer,
            criterion,
            metric_tracker,
            device,
            epoch,
        )
        valid_scores = valid_loop(
            model, valid_loader, criterion, metric_tracker, device
        )

        classes = ["Normal", "Shoplifting", "Doubt", "Background"]

        # --wandb--
        log_data = {
            "Epoch": epoch + 1,
            "train_loss": train_scores["loss"],
            "valid_loss": valid_scores["loss"],
        }

        for i, class_name in enumerate(classes[:-1]):
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

        wandb.log(log_data)
        wandb.sklearn.plot_confusion_matrix(
            train_scores["true"], train_scores["pred"], labels=classes
        )
        wandb.sklearn.plot_confusion_matrix(
            valid_scores["true"], valid_scores["pred"], labels=classes
        )

        # --save model--
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # epoch model 저장
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_path = tmp.name
            torch.save(model.state_dict(), model_path)

        artifact = wandb.Artifact(f"Epoch_{epoch+1}", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

        os.remove(model_path)

        # best model 저장
        if valid_scores["loss"] < best_loss:
            best_loss = valid_scores["loss"]
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch+1}\n"
            f"Train | f1 score: {train_scores['f1']} | precision: {train_scores['precision']} | recall: {train_scores['recall']} | loss: {train_scores['loss']}\n"
            f"Valid | f1 score: {valid_scores['f1']} | precision: {valid_scores['precision']} | recall: {valid_scores['recall']} | loss: {valid_scores['loss']}"
        )


if __name__ == "__main__":
    main()
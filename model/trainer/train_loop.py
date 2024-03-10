import torch
from tqdm import tqdm
from arch import FrameModel, ClipModel


def train_loop(
    model, train_loader, optimizer, criterion, metric_tracker, device, epoch
):
    model.to(device)
    model.train()

    for frames, labels in tqdm(
        train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}"
    ):
        frames, labels = frames.to(device), labels.to(device)

        if isinstance(model, FrameModel):
            labels = labels.view(-1)
        elif isinstance(model, ClipModel):
            labels = labels[:, -1]

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        metric_tracker.update(loss=loss.item(), true=labels, pred=preds)

    return metric_tracker.result()

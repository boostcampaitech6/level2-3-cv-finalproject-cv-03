import torch
from tqdm import tqdm


def train_loop(
    model,
    train_loader,
    optimizer,
    criterion,
    scheduler,
    metric_tracker,
    device,
    epoch,
):
    model.to(device)
    model.train()

    metric_tracker.reset()

    for frames, labels in tqdm(train_loader, desc=f"[Epoch {epoch} (Train)]"):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        metric_tracker.update(loss=loss.item(), true=labels, pred=preds)

    if scheduler is not None:
        scheduler.step()

    return metric_tracker.result()

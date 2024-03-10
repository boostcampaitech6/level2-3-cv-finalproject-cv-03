import torch
from tqdm import tqdm


def valid_loop(model, valid_loader, criterion, metric_tracker, device):
    model.to(device)
    model.eval()

    metric_tracker.reset()

    with torch.no_grad():
        for frames, labels in tqdm(valid_loader, desc="[Valid]"):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(frames)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()

            metric_tracker.update(loss=loss.item(), true=labels, pred=preds)

    return metric_tracker.result()

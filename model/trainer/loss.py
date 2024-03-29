import torch.nn as nn
import torch.nn.functional as F


class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7, weight=None):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

        if self.weight is not None:
            weight = self.weight.unsqueeze(0)
        else:
            weight = 1.0

        tp = (y_true * y_pred).sum(dim=0)  # True Positives
        fp = ((1 - y_true) * y_pred).sum(dim=0)  # False Positives
        fn = (y_true * (1 - y_pred)).sum(dim=0)  # False Negatives

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1_loss = 1 - f1

        loss = f1_loss * weight

        return loss.mean()


class FbetaLoss(nn.Module):
    def __init__(self, beta=1, epsilon=1e-7, weight=None):
        super(FbetaLoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

        if self.weight is not None:
            weight = self.weight.unsqueeze(0)
        else:
            weight = 1.0

        tp = (y_true * y_pred).sum(dim=0)  # True Positives
        fp = ((1 - y_true) * y_pred).sum(dim=0)  # False Positives
        fn = (y_true * (1 - y_pred)).sum(dim=0)  # False Negatives

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        fbeta = (
            (1 + self.beta**2)
            * (precision * recall)
            / ((self.beta**2 * precision) + recall + self.epsilon)
        )
        fbeta_loss = 1 - fbeta

        loss = fbeta_loss * weight

        return loss.mean()


class CrossF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7, weight=None):
        super(CrossF1Loss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

        if self.weight is not None:
            weight = self.weight.unsqueeze(0)
        else:
            weight = 1.0

        tp = (y_true * y_pred).sum(dim=0)  # True Positives
        fp = ((1 - y_true) * y_pred).sum(dim=0)  # False Positives
        fn = (y_true * (1 - y_pred)).sum(dim=0)  # False Negatives

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1_loss = 1 - f1

        f1_loss = (f1_loss * weight).mean()
        ce_loss = F.cross_entropy(y_pred, y_true, weight=weight)

        loss = (0.5 * f1_loss) + (0.5 * ce_loss)

        return loss

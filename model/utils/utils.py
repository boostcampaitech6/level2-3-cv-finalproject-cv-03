from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

CLASSES = ["Normal", "Abnormal"]

RNN_INPUT_SIZE = {
    "yolov8n": 256,
    "yolov8s": 512,
    "yolov8m": 768,
    "yolov8l": 1024,
    "yolov8x": 1280,
    "ResNet34": 512,
    "ResNet50": 2048,
    "MobileNet": 1280,
    "ResNeXt": 2048,
    "VGG16": 25088,
    "MobileNetV3S": 576,
    "MobileNetV3L": 960,
}


class MetricTracker:
    def __init__(self):
        self.loss = {"total": None, "count": None}
        self.true = None
        self.pred = None

        self.reset()

    def reset(self):
        self.loss["total"] = 0
        self.loss["count"] = 0
        self.true = []
        self.pred = []

    def update(self, loss=None, true=None, pred=None):
        assert all(
            x is not None for x in [loss, true, pred]
        ), "Warning: None exists."

        self.loss["total"] += loss
        self.loss["count"] += 1
        self.true.extend(true)
        self.pred.extend(pred)

    def result(self):
        scores = {
            "true": self.true,
            "pred": self.pred,
            "loss": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

        scores["loss"] = self.loss["total"] / self.loss["count"]
        scores["precision"] = precision_score(
            self.true, self.pred, average=None, zero_division=0.0
        )
        scores["recall"] = recall_score(
            self.true, self.pred, average=None, zero_division=0.0
        )
        # scores["f1"] = f1_score(
        #     self.true, self.pred, average=None, zero_division=0.0
        # )
        scores["f1"] = fbeta_score(
            self.true, self.pred, average=None, beta=1.5, zero_division=0.0
        )

        return scores


class LabelTracker:
    def __init__(self):
        self.labels = None
        self.probs = None

        self.reset()

    def reset(self):
        self.labels = {}
        self.probs = {}

    def update(self, clip_names, labels, probs):
        for clip_name, label, prob in zip(clip_names, labels, probs):
            self.labels[clip_name] = label
            self.probs[clip_name] = prob

    def result(self):
        return self.labels, self.probs

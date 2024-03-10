from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


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
        scores["f1"] = f1_score(
            self.true, self.pred, average=None, zero_division=0.0
        )

        return scores


class LabelTracker:
    def __init__(self):
        self.labels = None

        self.reset()

    def reset(self):
        self.labels = {}

    def update(self, file_names, labels):
        for file_name, label in zip(file_names, labels):
            self.labels[file_name] = label

    def result(self):
        return self.labels

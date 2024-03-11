import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetGRU(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AvgPool2d((7, 7))

        gru_hidden_size = 512
        self.gru = nn.GRU(
            1280,
            gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x):
        batch, frame, channel, height, width = x.size()
        x = x.view(batch * frame, channel, height, width)

        cnn_out = self.mobilenet(x)
        cnn_out = self.avgpool(cnn_out)
        cnn_out = cnn_out.view(batch, frame, -1)

        rnn_out, _ = self.gru(cnn_out)
        # rnn_out = rnn_out.contiguous().view(batch * frame, -1)
        rnn_out = rnn_out[:, -1, :]

        out = self.fc(rnn_out)
        out = F.softmax(out, dim=1)

        return out

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FrameModel(nn.Module):
    def __init__(self, cnn, rnn, hidden=512, nc=2):
        super(FrameModel, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.fc = nn.Linear(hidden, nc)

    def forward(self, x):
        batch, frame, C, H, W = x.shape
        x = x.view(batch * frame, C, H, W)

        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch, frame, -1)

        rnn_out = self.rnn(cnn_out)
        rnn_out = rnn_out.contiguous().view(batch * frame, -1)

        class_out = self.fc(rnn_out)

        return class_out


class ClipModel(nn.Module):
    def __init__(self, cnn, rnn, hidden=512, nc=2):
        super(ClipModel, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.fc = nn.Linear(hidden, nc)

    def forward(self, x):
        batch, frame, C, H, W = x.shape
        x = x.view(batch * frame, C, H, W)

        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch, frame, -1)

        rnn_out = self.rnn(cnn_out)
        rnn_out = rnn_out[:, -1, :]

        class_out = self.fc(rnn_out)

        return class_out


class CNNRNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNRNN, self).__init__()

        # CNN
        self.features = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # RNN
        rnn_in_features = 1280  # cnn out_features
        rnn_hidden_size = 256
        rnn_num_layers = 2
        bidirectional = False
        self.rnn = nn.GRU(
            rnn_in_features,
            rnn_hidden_size,
            rnn_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2,
        )

        # FCNN
        fcnn_in_features = rnn_hidden_size
        if bidirectional:
            fcnn_in_features = rnn_hidden_size * 2

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(fcnn_in_features, num_classes)
        )

    def forward(self, x):
        batch, frame, C, H, W = x.shape
        x = x.view(batch * frame, C, H, W)

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch, frame, -1)

        x, _ = self.rnn(x)
        x = x[:, -1, :]

        x = self.fc(x)
        return x


class CNNRNNAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNRNNAttention, self).__init__()

        # CNN
        self.features = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # RNN
        rnn_in_features = 1280  # cnn out_features
        rnn_hidden_size = 256
        rnn_num_layers = 2
        bidirectional = False
        self.rnn = nn.GRU(
            rnn_in_features,
            rnn_hidden_size,
            rnn_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2,
        )

        # FCNN + Attention
        fcnn_in_features = rnn_hidden_size
        if bidirectional:
            fcnn_in_features = rnn_hidden_size * 2

        self.attention = nn.Sequential(
            nn.Linear(fcnn_in_features, 1), nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(fcnn_in_features, num_classes)
        )

    def forward(self, x):
        batch, frame, C, H, W = x.size()
        x = x.view(batch * frame, C, H, W)

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch, frame, -1)

        x, _ = self.rnn(x)

        attn_weights = self.attention(x)
        x = (x * attn_weights).sum(dim=1)

        x = self.fc(x)
        return x


class CNNFCNN(nn.Module):
    def __init__(self, frame=12, num_classes=2):
        super(CNNFCNN, self).__init__()

        # CNN
        self.cnn = models.mobilenet_v2(weights="DEFAULT")
        self.cnn.classifier[-1] = nn.Linear(1280, num_classes)

        # FCNN
        fcnn_in_features = frame * num_classes
        self.fc = nn.Sequential(
            nn.Linear(fcnn_in_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        batch, frame, C, H, W = x.size()
        x = x.view(batch * frame, C, H, W)

        x = self.cnn(x)

        x = F.softmax(x, dim=1)
        x = x.view(batch, -1)

        x = self.fc(x)
        return x


class CNNAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNAttention, self).__init__()

        # CNN
        self.features = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        batch, frame, C, H, W = x.size()
        x = x.view(batch * frame, C, H, W)

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch, frame, -1)

        attn_weights = self.attention(x)
        x = (x * attn_weights).sum(dim=1)

        x = self.fc(x)
        return x


class CNN1DConv(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN1DConv, self).__init__()

        # CNN
        self.features = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 1DConv
        in_channels = 1280  # cnn out_features
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=512, out_channels=256, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

        # FCNN
        fcnn_in_features = 256
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(fcnn_in_features, num_classes)
        )

    def forward(self, x):
        batch, frame, C, H, W = x.size()
        x = x.view(batch * frame, C, H, W)

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch, frame, -1)
        x = x.transpose(1, 2)

        x = self.conv1d(x)
        x = x[:, :, -1]

        x = self.fc(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super(TemporalBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, in_features, channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = in_features if i == 0 else channels[i - 1]
            out_channels = channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size // 2,
                    dropout=dropout,
                )
            ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNTCN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNTCN, self).__init__()

        # CNN
        self.features = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # TCN
        tcn_in_features = 1280  # out_features
        tcn_channels = [256, 128]
        self.tcn = TemporalConvNet(tcn_in_features, tcn_channels)

        # FCNN
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(tcn_channels[-1], num_classes)
        )

    def forward(self, x):
        batch, frames, C, H, W = x.size()
        x = x.view(batch * frames, C, H, W)

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch, frames, -1).transpose(1, 2)

        x = self.tcn(x)
        x = x[:, :, -1]

        x = self.fc(x)
        return x

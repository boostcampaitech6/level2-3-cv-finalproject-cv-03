import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, cnn, hidden_size, num_layers):
        super(GRU, self).__init__()

        if cnn == "YOLO_n":
            input_size = 256
        elif cnn == "ResNet32" or cnn == "YOLO_s":
            input_size = 512
        elif cnn == "YOLO_m":
            input_size = 768
        elif cnn == "YOLO_l":
            input_size = 1024
        elif cnn == "MobileNet" or cnn == "YOLO_x":
            input_size = 1280
        elif cnn == "ResNet50" or cnn == "ResNeXt":
            input_size = 2048
        elif cnn == "VGG16":
            input_size = 25088

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        x, _ = self.gru(x)
        return x


class LSTM(nn.Module):
    def __init__(self, cnn, hidden_size, num_layers):
        super(LSTM, self).__init__()

        if cnn == "YOLO_n":
            input_size = 256
        elif cnn == "ResNet32" or cnn == "YOLO_s":
            input_size = 512
        elif cnn == "YOLO_m":
            input_size = 768
        elif cnn == "YOLO_l":
            input_size = 1024
        elif cnn == "MobileNet" or cnn == "YOLO_x":
            input_size = 1280
        elif cnn == "ResNet50" or cnn == "ResNeXt":
            input_size = 2048
        elif cnn == "VGG16":
            input_size = 25088

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

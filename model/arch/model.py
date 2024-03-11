import torch.nn as nn


class FrameModel(nn.Module):
    def __init__(self, cnn, rnn, hidden=512, nc=4):
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
    def __init__(self, cnn, rnn, hidden=512, nc=4):
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

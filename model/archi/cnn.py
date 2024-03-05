import torch.nn as nn
from torchvision import models
from ultralytics.nn.tasks import ClassificationModel


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AvgPool2d((7, 7))

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.avgpool(x)
        return x


class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.yolo = ClassificationModel(
            cfg="/data/ephemeral/home/yolo_yaml/yolov8x-cls.yaml", ch=3, nc=2
        )

    def forward(self, x):
        x = self.yolo(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(weights="DEFAULT").features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.vgg16(x)
        x = self.avgpool(x)
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        resnet = models.resnet34(weights="DEFAULT")
        self.feature = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(weights="DEFAULT")
        self.feature = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        return x


class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt, self).__init__()
        resnext = models.resnext50_32x4d(weights="DEFAULT")
        self.feature = nn.Sequential(*list(resnext.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        return x

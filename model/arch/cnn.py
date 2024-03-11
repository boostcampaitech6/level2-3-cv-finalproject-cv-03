import os
import torch.nn as nn
from torchvision import models
from ultralytics.nn.tasks import ClassificationModel


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights="DEFAULT").features
        self.avgpool = nn.AvgPool2d((7, 7))

    def __str__(self):
        return "MobileNet"

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.avgpool(x)
        return x


class YOLO(nn.Module):
    def __init__(self, cfg="./yolo_yaml/yolov8n-cls.yaml"):
        super(YOLO, self).__init__()
        self.cfg = cfg
        self.yolo = ClassificationModel(cfg=self.cfg, ch=3, nc=4)

    def __str__(self):
        model_name = os.path.basename(self.cfg).split(".")[0].split("-")[0]
        return model_name

    def forward(self, x):
        x = self.yolo(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(weights="DEFAULT").features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def __str__(self):
        return "VGG16"

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

    def __str__(self):
        return "ResNet34"

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

    def __str__(self):
        return "ResNet50"

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

    def __str__(self):
        return "ResNeXt"

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        return x

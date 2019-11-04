# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models

class_num = 10  # Maybe impoted from other config file


class ClassifierModel(object):

    def __init__(self, model_name, method="TL"):
        if method == "BL":
            pretrained = False
        else:
            pretrained = True

        self.input_size = 224

        print("loading model...")
        if model_name[:6] == "resnet":
            self.model = eval("models.{}({})".format(model_name, pretrained))
            """ Resnet相关
            可为： 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'

            """
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.fc.in_features
            self.model.fc = nn.Linear(num_fc_in, class_num)  # 修改最后一层(fc层)

        elif model_name == "alexnet":
            """ Alexnet
            只能为: 'alexnet'

            """
            self.model = models.alexnet(pretrained)
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_fc_in, class_num)

        elif "vgg" in model_name:
            """ VGG相关：
            可为：'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'

            """
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_fc_in, class_num)

        elif "squeezenet" in model_name:
            """ Squeezenet相关
            可为： 'squeezenet1_0', 'squeezenet1_1'

            """
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier[1] = nn.Conv2d(512, class_num, kernel_size=(1, 1), stride=(1, 1))
            self.model.num_classes = class_num

        elif "densenet" in model_name:
            """ Densenet相关
            可为： 'densenet121', 'densenet161', 'densenet169', 'densenet201',
            """
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_fc_in, class_num)

        elif "inception" in model_name:
            """Inception相关
            可为： 'inception_v3'

            """
            self.input_size = 299
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_fc_in, class_num)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, class_num)

        elif "mobilenet" in model_name:
            """ mobilenet相关
            可为: 'mobilenet_v2'

            """
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_fc_in, class_num)

        elif "mnasnet" in model_name:
            """ mnasnet相关
            可为: 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'

            """
            if model_name == "mnasnet1_3" and pretrained:
                print("No checkpoint is available for model type mnasnet1_3")
                print("change the method to 'BL'")
                pretrained = False
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_fc_in, class_num)

        elif "googlenet" in model_name:
            """ googlenet相关
            可为: 'googlenet'

            """
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.fc.in_features
            self.model.fc = nn.Linear(num_fc_in, class_num)

        elif "resnext" in model_name:
            """ resnext相关
            可为: 'resnext50_32x4d', 'resnext101_32x8d'

            """
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.fc.in_features
            self.model.fc = nn.Linear(num_fc_in, class_num)

        elif "shufflenet" in model_name:
            """ shufflenet相关
            可为: 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
                 'shufflenet_v2_x2_0'

            """
            if model_name == "shufflenet_v2_x2_0":
                print("pretrained shufflenetv2_x2.0 is not supported as of now, use 'BL' method...")
                pretrained = False
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.fc.in_features
            self.model.fc = nn.Linear(num_fc_in, class_num)

        elif "wide_resnet" in model_name:
            """ wide_resnet相关
            可为: 'wide_resnet50_2', 'wide_resnet101_2'

            """
            self.model = eval("models.{}({})".format(model_name, pretrained))
            if method == "TF":
                for param in self.model.parameters():
                    param.requires_grad = False
            num_fc_in = self.model.fc.in_features
            self.model.fc = nn.Linear(num_fc_in, class_num)

        elif model_name == "selfdefine":
            """ 自行定义的网络
            只能为: 'selfdefine'

            """
            self.model = SeflNet(class_num)

        else:
            sys.exit("Invalid model name, exiting...")
        print("model loaded.")


# You can define by yourself
class SeflNet(nn.Module):
    def __init__(self, class_num):
        super(SeflNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(3, 3),
                               padding=(1, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2, 2), stride=(7, 7),
                               padding=(1, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'alexnet', 'vgg11',
                   'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                   'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet161', 'densenet169',
                   'densenet201', 'inception_v3', 'mobilenet_v2', 'mnasnet0_5',
                   'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'googlenet', 'resnext50_32x4d',
                   'resnext101_32x8d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
                   'shufflenet_v2_x2_0', 'wide_resnet50_2', 'wide_resnet101_2', 'selfdefine']
    methods = ["BL", "TL", "FT"]
    import random
    import torch
    while True:
        model_name = model_names[random.randint(0, len(model_names)-1)]
        method = methods[random.randint(0, len(methods)-1)]
        class_num = random.randint(1, 1000)
        print(f"model_name: {model_name}; method: {method}; class_num: {class_num}")
        model = ClassifierModel(model_name, method=method)
        batch = random.randint(1, 3)
        img = torch.randn(batch, 3, model.input_size, model.input_size)
        print(f"input size: {img.size()}")
        output = model.model(img)
        if model_name in ["googlenet", 'inception_v3']:
            print(f"output size: {[o.size() for o in output]}")
        else:
            print(f"output size: {output.size()}")

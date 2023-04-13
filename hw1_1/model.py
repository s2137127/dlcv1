import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.n = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #             nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #             nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #             nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #             nn.Conv2d(512, 512, kernel_size=3, stride=1),
            #             nn.BatchNorm2d(512),
            #             nn.ReLU(),
            #             nn.MaxPool2d(2, stride=1),

            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #             nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #             nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=18432, out_features=4096, bias=True),
            # nn.Dropout(p=0.2,inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            # nn.Dropout(p=0.2,inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=50, bias=True),
        )

    def forward(self, x):
        x = self.n(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output

    def name(self):
        return "MyNet"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('device will be used as ', device)

# conv size has been hard coded
i1 = 32
i2 = 64
i3 = 128
i4 = 256
i5 = 512
class simplegalaxy(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=i1, kernel_size=3, padding=1),  # 3, 32x32 -> 32, 32x32
            nn.BatchNorm2d(i1),
            nn.ReLU(),
            nn.Conv2d(in_channels=i1, out_channels=i2, kernel_size=3, padding=1),  # 3, 32x32 -> 32, 32x32
            nn.BatchNorm2d(i2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32, 32x32 -> 32, 16x16
        )

        self.layer2 = nn.Sequential(

            nn.Conv2d(in_channels=i2, out_channels=i3, kernel_size=3, padding=1),  # 32, 16x16 -> 64, 16x16
            nn.BatchNorm2d(i3),
            nn.ReLU(),
            nn.Conv2d(in_channels=i3, out_channels=i4, kernel_size=3, padding=1),  # 32, 16x16 -> 64, 16x16
            nn.BatchNorm2d(i4),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64, 16x16 -> 64, 8x8
        )

        self.layer3 = nn.Sequential(

            nn.Conv2d(in_channels=i4, out_channels=i5, kernel_size=3, padding=1),  # 32, 16x16 -> 64, 16x16
            nn.BatchNorm2d(i5),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64, 16x16 -> 64, 8x8
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(i5 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, numclasses),            # 32 * 32 * 32 -> numclasses


        )

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = x.view(x.size(0), -1)  # flatten

        x = self.fc(x)

        return x


    
    


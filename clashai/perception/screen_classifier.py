import torch.nn as nn


class MyCustomCNN(nn.Module):
    """
    CNN personnalisé pour la classification dans Clash of Clans.
    Supporte n'importe quelle taille d'image en entrée grâce à AdaptiveAvgPool2d.
    Utilisé pour : bâtiments (128x128) et états d'écran (224x224).
    """

    def __init__(self, num_classes: int):
        super(MyCustomCNN, self).__init__()

        # Bloc convolutif 1 : 3 -> 32 canaux
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Bloc convolutif 2 : 32 -> 64 canaux
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bloc convolutif 3 : 64 -> 128 canaux
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # AdaptiveAvgPool : force la sortie à 8x8 quelle que soit la taille d'entrée
        # 128x128 → après 3 pools → 16x16 → adaptive → 8x8
        # 224x224 → après 3 pools → 28x28 → adaptive → 8x8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Couches fully connected
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(128 * 8 * 8, 512)   # 8192 → 512
        self.relu4   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
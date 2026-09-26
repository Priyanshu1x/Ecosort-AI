import torch
import torch.nn as nn
import torchvision.models as models


# BASELINE CNN

class BaselineCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(

            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.LazyLinear(128),
            nn.ReLU(),

            nn.Linear(128, 4)
        )

    def forward(self, input_feature):

        x = self.feature(input_feature)
        x = self.classifier(x)

        return x


# RESNET18 TRANSFER LEARNING

def build_resnet18(num_classes=4):

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Replace the original 1000-class output layer
    model.fc = nn.Linear(512, num_classes)

    # Freeze the pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Train only the new classification layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def build_EfficientNet(num_classes=4):
    model=models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    model.classifier[1]=nn.Linear(1280,num_classes)

    for param in model.parameters():
        param.requires_grad=False

    for param in model.classifier.parameters():
        param.requires_grad=True

    return model
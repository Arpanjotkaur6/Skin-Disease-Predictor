import torch
import torch.nn as nn
from torchvision import models

class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseModel, self).__init__()

        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Get number of features
        num_features = self.backbone.classifier[1].in_features

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # Severity head
        self.severity_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        features     = self.backbone(x)
        disease_out  = self.disease_head(features)
        severity_out = self.severity_head(features)
        return disease_out, severity_out

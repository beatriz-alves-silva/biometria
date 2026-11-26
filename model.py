import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MobileNetV3Iris(nn.Module):
    def __init__(self, backbone_name="mobilenet_v3_large", pretrained=True, embedding_dim=128, freeze_backbone=False):
        super().__init__()
        if backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            in_features = base.classifier[0].in_features
        else:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)
            in_features = base.classifier[0].in_features

        self.features = base.features
        self.avgpool = base.avgpool
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Cabeça de extração de embedding biométrico
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.embedding_head(x)
        # Normalização L2 para matching por similaridade de cosseno
        return F.normalize(emb, p=2, dim=1)

class CosFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, w_norm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output
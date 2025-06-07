import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig

# 🔹 **Hybrid Model (EfficientNet + ViT)**
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()

        # Load Pretrained EfficientNet
        self.efficientnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.efficientnet.classifier = nn.Identity()  # Remove last FC layer

        # Load Pretrained Vision Transformer (ViT)
        self.vit_config = ViTConfig(image_size=224, num_labels=2)
        self.vit = ViTModel(self.vit_config)

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(1280 + self.vit_config.hidden_size, 512),  
            nn.Mish(),  # Activation function
            nn.Dropout(0.3),  
            nn.Linear(512, 2)  # Output layer (2 classes: Glaucoma, Normal)
        )

    def forward(self, x):
        eff_features = self.efficientnet(x)
        vit_features = self.vit(x).last_hidden_state[:, 0, :]
        features = torch.cat((eff_features, vit_features), dim=1)
        return self.fc(features)


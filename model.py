import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the Multi-Scale CNN
def create_cnn():
    cnn = resnet18(pretrained=True)
    cnn.fc = nn.Identity()  # Remove the final fully connected layer
    return cnn

# Define the Swin Transformer
def create_swin_transformer():
    swin_transformer = swin_tiny_patch4_window7_224(pretrained=True)
    swin_transformer.head = nn.Identity()  # Remove the classification head
    return swin_transformer

# Define the Multi-Scale CNN + Swin Transformer Model
class MultiScaleCNNSwinTransformer(nn.Module):
    def __init__(self):
        super(MultiScaleCNNSwinTransformer, self).__init__()
        self.cnn = create_cnn()
        self.swin_transformer = create_swin_transformer()
        self.fc = nn.Linear(768 + 512, 3)  # Combine CNN and Swin Transformer outputs

    def forward(self, x):
        # Debugging shape
        print(f"Input shape: {x.shape}")
        
        cnn_features = self.cnn(x)  # [batch_size, 512]
        print(f"CNN features shape: {cnn_features.shape}")

        swin_features = self.swin_transformer.forward_features(x)  # Use correct forward method
        swin_features = swin_features.mean(dim=[1, 2])  # Global average pooling to [batch_size, 768]
        print(f"Swin Transformer features shape after pooling: {swin_features.shape}")

        combined_features = torch.cat([cnn_features, swin_features], dim=1)
        print(f"Combined features shape: {combined_features.shape}")

        out = self.fc(combined_features)
        print(f"Output shape: {out.shape}")
        return out

# Dataset Preparation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}




# Initialize model, criterion, and optimizer
model = MultiScaleCNNSwinTransformer()


import torch
import torch.nn as nn
from torchvision import models

def get_skin_model(num_classes=7):
    # Using MobileNetV3: High performance, efficient for 2026 standards
    model = models.mobilenet_v3_large(weights='DEFAULT')
    
    # Freeze layers to keep pre-trained features
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
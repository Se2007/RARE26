import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def resnet50(pretrained=True):
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50()
        
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Linear(num_ftrs, 1)
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True).to(device)

    model.eval()
    print(f"Model successfully loaded on {device}!")
    print(model.fc) 

    dummy_batch = torch.randn(16, 3, 224, 224).to(device)

    with torch.no_grad():
        logits = model(dummy_batch)

    print(f"Logits shape: {logits.shape}")
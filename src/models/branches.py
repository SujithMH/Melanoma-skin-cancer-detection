import torch.nn as nn
import timm

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # EfficientNet-B1 is lightweight and highly accurate
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class BaselineViT(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # DeiT-Small is a data-efficient Vision Transformer that fits in 4GB VRAM
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor of class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class HybridLossFocal(nn.Module):
    def __init__(self, alpha_cls=None, gamma=2.0, sev_weight=0.428):
        super().__init__()
        self.cls_criterion = FocalLoss(alpha=alpha_cls, gamma=gamma)
        self.sev_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.sev_weight = sev_weight
        
    def forward(self, cls_preds, sev_preds, cls_labels, sev_labels):
        loss_cls = self.cls_criterion(cls_preds, cls_labels)
        loss_sev = self.sev_criterion(sev_preds, sev_labels)
        
        total_loss = loss_cls + (self.sev_weight * loss_sev)
        return total_loss, loss_cls, loss_sev
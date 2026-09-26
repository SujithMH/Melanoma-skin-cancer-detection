import torch
import torch.nn as nn
import timm

class HybridDeepNet(nn.Module):
    def __init__(self, num_classes=7, num_sev_tiers=4, fusion_type='concat', fc_hidden=512, dropout=0.3):
        super().__init__()
        self.fusion_type = fusion_type
        
        self.cnn = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)
        self.vit = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0)
        
        cnn_dim = self.cnn.num_features # 1280
        vit_dim = self.vit.num_features # 384
        
        if self.fusion_type == 'concat':
            fused_dim = cnn_dim + vit_dim
        elif self.fusion_type == 'add':
            self.cnn_proj = nn.Linear(cnn_dim, fc_hidden)
            self.vit_proj = nn.Linear(vit_dim, fc_hidden)
            fused_dim = fc_hidden
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
            
        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.cls_head = nn.Linear(fc_hidden, num_classes)
        self.sev_head = nn.Linear(fc_hidden, num_sev_tiers)

    def forward(self, x):
        f_cnn = self.cnn(x)
        f_vit = self.vit(x)
        
        if self.fusion_type == 'concat':
            fused = torch.cat((f_cnn, f_vit), dim=1)
        elif self.fusion_type == 'add':
            fused = self.cnn_proj(f_cnn) + self.vit_proj(f_vit)
            
        shared = self.shared_fc(fused)
        cls_out = self.cls_head(shared)
        sev_out = self.sev_head(shared)
        
        return cls_out, sev_out
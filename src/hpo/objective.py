import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from src.data.dataset import MelanomaDataset
from src.models.hybrid import HybridDeepNet
from src.models.losses import HybridLoss
from src.hpo.encoding import decode_vector

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_candidate(vector, seed=42, proxy_epochs=5, proxy_fraction=0.35, device='cuda'):
    set_seed(seed)
    config = decode_vector(vector)
    
    # Load dataset & create proxy subset
    full_train = MelanomaDataset('splits/train.csv', 'data/cache', is_train=True)
    num_samples = int(len(full_train) * proxy_fraction)
    
    subset_indices = list(range(num_samples))
    proxy_train = Subset(full_train, subset_indices)
    
    train_loader = DataLoader(proxy_train, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = MelanomaDataset('splits/val.csv', 'data/cache', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    model = HybridDeepNet(
        fusion_type=config['fusion_type'],
        fc_hidden=config['fc_hidden'],
        dropout=config['dropout']
    ).to(device)
    
    # Separate learning rates for backbone vs head
    backbone_params = list(model.cnn.parameters()) + list(model.vit.parameters())
    head_params = [p for n, p in model.named_parameters() if not any(p is bp for bp in backbone_params)]
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': config['head_lr'] * config['backbone_lr_mult']},
        {'params': head_params, 'lr': config['head_lr']}
    ], weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=proxy_epochs)
    criterion = HybridLoss(sev_weight=config['sev_weight'])
    scaler = torch.amp.GradScaler('cuda')
    
    best_macro_f1 = 0.0
    
    for epoch in range(proxy_epochs):
        model.train()
        for images, cls_labels, sev_labels in train_loader:
            images = images.to(device)
            cls_labels, sev_labels = cls_labels.to(device), sev_labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                cls_preds, sev_preds = model(images)
                loss, _, _ = criterion(cls_preds, sev_preds, cls_labels, sev_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        
        # Validation check
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, cls_labels, _ in val_loader:
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    cls_preds, _ = model(images)
                preds = torch.argmax(cls_preds, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(cls_labels.numpy())
                
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            
    # Cleanup memory
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    
    return float(best_macro_f1)
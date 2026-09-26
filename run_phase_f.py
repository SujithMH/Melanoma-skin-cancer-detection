import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import time

from src.data.dataset import MelanomaDataset
from src.models.hybrid import HybridDeepNet
from src.models.losses import HybridLoss

def train_final_model(device, epochs=20):
    # The winning hyperparameters
    config = {
        'head_lr': 9.8e-4,
        'backbone_lr_mult': 0.14,
        'weight_decay': 0.01,
        'dropout': 0.357,
        'fc_hidden': 448,
        'fusion_type': 'concat',
        'sev_weight': 0.428
    }
    
    print("\n--- Starting Final Optimized Training ---")
    print(f"Config: {config}")
    
    # 100% of data used here
    train_dataset = MelanomaDataset('splits/train.csv', 'data/cache', is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = MelanomaDataset('splits/val.csv', 'data/cache', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    model = HybridDeepNet(
        fusion_type=config['fusion_type'],
        fc_hidden=config['fc_hidden'],
        dropout=config['dropout']
    ).to(device)
    
    backbone_params = list(model.cnn.parameters()) + list(model.vit.parameters())
    head_params = [p for n, p in model.named_parameters() if not any(p is bp for bp in backbone_params)]
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': config['head_lr'] * config['backbone_lr_mult']},
        {'params': head_params, 'lr': config['head_lr']}
    ], weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = HybridLoss(sev_weight=config['sev_weight'])
    scaler = torch.amp.GradScaler('cuda')
    
    best_macro_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
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
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation Phase
        model.eval()
        all_cls_preds = []
        all_cls_labels = []
        
        with torch.no_grad():
            for images, cls_labels, sev_labels in val_loader:
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    cls_preds, _ = model(images)
                
                preds = torch.argmax(cls_preds.float(), dim=1)
                all_cls_preds.extend(preds.cpu().numpy())
                all_cls_labels.extend(cls_labels.numpy())
                
        acc = accuracy_score(all_cls_labels, all_cls_preds)
        macro_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {time.time() - start_time:.0f}s | "
              f"T-Loss: {train_loss/len(train_loader):.4f} | V-Acc: {acc:.4f} | V-Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            # Save the final optimized weights
            torch.save(model.state_dict(), 'best_optimized_hybrid.pth')
            
    print(f"\nFinal Optimized Best Macro-F1: {best_macro_f1:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_final_model(device)
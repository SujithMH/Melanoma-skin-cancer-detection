import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import time

from src.data.dataset import MelanomaDataset
from src.models.hybrid import HybridDeepNet
from src.models.losses import HybridLoss

def train_hybrid(model, train_loader, val_loader, device, epochs=15):
    criterion = HybridLoss(sev_weight=0.5)
    
    # We use a lower learning rate than the baselines because the backbones are already adapted
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    best_macro_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_cls_loss, train_sev_loss = 0.0, 0.0, 0.0
        start_time = time.time()
        
        for images, cls_labels, sev_labels in train_loader:
            images = images.to(device)
            cls_labels, sev_labels = cls_labels.to(device), sev_labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                cls_preds, sev_preds = model(images)
                loss, cls_loss, sev_loss = criterion(cls_preds, sev_preds, cls_labels, sev_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_sev_loss += sev_loss.item()
            
        scheduler.step()
        
        # Validation Phase
        model.eval()
        all_cls_preds = []
        all_cls_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, cls_labels, sev_labels in val_loader:
                images = images.to(device)
                cls_labels, sev_labels = cls_labels.to(device), sev_labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    cls_preds, sev_preds = model(images)
                    loss, _, _ = criterion(cls_preds, sev_preds, cls_labels, sev_labels)
                    
                val_loss += loss.item()
                preds = torch.argmax(cls_preds, dim=1)
                
                all_cls_preds.extend(preds.cpu().numpy())
                all_cls_labels.extend(cls_labels.cpu().numpy())
                
        acc = accuracy_score(all_cls_labels, all_cls_preds)
        macro_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro')
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.0f}s | "
              f"T-Loss: {train_loss/len(train_loader):.4f} (Cls: {train_cls_loss/len(train_loader):.4f}, Sev: {train_sev_loss/len(train_loader):.4f}) | "
              f"V-Loss: {val_loss/len(val_loader):.4f} | "
              f"V-Acc: {acc:.4f} | V-Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            
    return best_macro_f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Start with batch_size=16. If it crashes with CUDA OOM, lower to 8.
    train_dataset = MelanomaDataset('splits/train.csv', 'data/cache', is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = MelanomaDataset('splits/val.csv', 'data/cache', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print("\n--- Training Hybrid Model (Concat Fusion) ---")
    model = HybridDeepNet(fusion_type='concat').to(device)
    hybrid_f1 = train_hybrid(model, train_loader, val_loader, device, epochs=15)
    
    print("\n======================================")
    print("PHASE C RESULTS")
    print(f"Hybrid Model Best Macro-F1: {hybrid_f1:.4f}")
    print(f"(Target to beat: 0.6981)")
    print("======================================")

if __name__ == "__main__":
    main()
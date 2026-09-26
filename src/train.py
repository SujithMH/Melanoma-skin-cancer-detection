import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import time

def train_model(model, train_loader, val_loader, device, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    best_macro_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for images, cls_labels, _ in train_loader:
            images, cls_labels = images.to(device), cls_labels.to(device)
            
            optimizer.zero_grad()
            
            # AMP Context Manager for 4GB VRAM survival
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, cls_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation Phase
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, cls_labels, _ in val_loader:
                images, cls_labels = images.to(device), cls_labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, cls_labels)
                    
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.0f}s | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {acc:.4f} | Val Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            # In a full run we would save the weights here. For baselines, we just want the number.
            
    return best_macro_f1
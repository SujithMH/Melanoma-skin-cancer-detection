import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.data.dataset import MelanomaDataset
from src.models.hybrid import HybridDeepNet

def generate_report():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating full evaluation report using device: {device}")
    
    config = {
        'fc_hidden': 448,
        'fusion_type': 'concat',
        'dropout': 0.357
    }
    
    model = HybridDeepNet(
        fusion_type=config['fusion_type'],
        fc_hidden=config['fc_hidden'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load('best_optimized_hybrid_focal.pth', map_location=device))
    model.eval()
    
    val_dataset = MelanomaDataset('splits/val.csv', 'data/cache', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    all_cls_preds, all_cls_labels = [], []
    all_sev_preds, all_sev_labels = [], []
    
    with torch.no_grad():
        for images, cls_labels, sev_labels in val_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda'):
                cls_preds, sev_preds = model(images)
                
            c_preds = torch.argmax(cls_preds.float(), dim=1).cpu().numpy()
            s_preds = torch.argmax(sev_preds.float(), dim=1).cpu().numpy()
            
            all_cls_preds.extend(c_preds)
            all_cls_labels.extend(cls_labels.numpy())
            
            # Mask out vasc lesions (-1) for severity evaluation
            valid_mask = (sev_labels.numpy() != -1)
            all_sev_preds.extend(s_preds[valid_mask])
            all_sev_labels.extend(sev_labels.numpy()[valid_mask])

    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # 1. Print Per-Class Classification Report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT (PRIMARY LESION HEAD)")
    print("="*60)
    print(classification_report(all_cls_labels, all_cls_preds, target_names=class_names, digits=4))
    
    # 2. Severity Accuracy
    sev_acc = accuracy_score(all_sev_labels, all_sev_preds)
    sev_f1 = f1_score(all_sev_labels, all_sev_preds, average='macro')
    print("="*60)
    print(f"SECONDARY SEVERITY HEAD ACCURACY: {sev_acc*100:.2f}% | Macro-F1: {sev_f1:.4f}")
    print("="*60)
    
    # 3. Plot Normalized Confusion Matrix
    cm = confusion_matrix(all_cls_labels, all_cls_preds, normalize='true')
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix — Hybrid XAI DeepNet')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\nSaved confusion matrix plot to 'confusion_matrix.png'")

if __name__ == "__main__":
    generate_report()
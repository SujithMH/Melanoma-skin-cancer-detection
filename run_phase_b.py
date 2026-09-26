import torch
from torch.utils.data import DataLoader
from src.data.dataset import MelanomaDataset
from src.models.branches import BaselineCNN, BaselineViT
from src.train import train_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Adjust num_workers to 0 if Windows multiprocessing freezes
    train_dataset = MelanomaDataset('splits/train.csv', 'data/cache', is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = MelanomaDataset('splits/val.csv', 'data/cache', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print("\n--- Training EfficientNet-B1 Baseline ---")
    cnn_model = BaselineCNN().to(device)
    cnn_f1 = train_model(cnn_model, train_loader, val_loader, device, epochs=15)
    
    # Free up VRAM before loading the next model
    del cnn_model
    torch.cuda.empty_cache()

    print("\n--- Training DeiT-Small Baseline ---")
    vit_model = BaselineViT().to(device)
    vit_f1 = train_model(vit_model, train_loader, val_loader, device, epochs=15)

    print("\n======================================")
    print("PHASE B RESULTS")
    print(f"EfficientNet-B1 Best Macro-F1: {cnn_f1:.4f}")
    print(f"DeiT-Small Best Macro-F1:      {vit_f1:.4f}")
    print("======================================")

if __name__ == "__main__":
    main()
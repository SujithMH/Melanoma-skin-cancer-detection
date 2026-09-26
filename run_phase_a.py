import time
import torch
from torch.utils.data import DataLoader
from src.data.dataset import MelanomaDataset

def test_dataloader():
    print("Testing DataLoader performance...")
    
    train_dataset = MelanomaDataset(
        csv_file='splits/train.csv', 
        img_dir='data/cache', 
        is_train=True
    )
    
    # batch_size=16 for the 4GB VRAM constraint, num_workers=4 speeds up IO
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    start_time = time.time()
    
    for batch_idx, (images, cls_labels, sev_labels) in enumerate(train_loader):
        # Simulate moving to GPU to ensure no weird dtype issues
        if torch.cuda.is_available():
            images = images.cuda()
            cls_labels = cls_labels.cuda()
            sev_labels = sev_labels.cuda()
            
        if batch_idx % 50 == 0:
            print(f"Loaded batch {batch_idx}/{len(train_loader)}... Shape: {images.shape}")
            
    end_time = time.time()
    print(f"One full epoch of dataloading took {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    test_dataloader()
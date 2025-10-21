import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc  # Để dọn bộ nhớ
from torch.cuda.amp import autocast, GradScaler  # Để tối ưu bộ nhớ

# Cấu hình seed cho tính ổn định
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Tạo thư mục dataset với cấu trúc train/val
def prepare_dataset(src_root, dst_root, train_ratio=0.8):
    # Tạo cấu trúc thư mục
    splits = ['train', 'val']
    for split in splits:
        for disease in ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']:
            os.makedirs(os.path.join(dst_root, split, disease), exist_ok=True)
    
    # Duyệt qua từng loại bệnh và chia dữ liệu
    for disease in ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']:
        src_path = os.path.join(src_root, disease)
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} không tồn tại")
            continue
            
        # Lấy danh sách ảnh
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Chia train/val
        train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
        
        # Copy ảnh vào thư mục tương ứng
        for img in train_images:
            src_file = os.path.join(src_path, img)
            dst_file = os.path.join(dst_root, 'train', disease, img)
            shutil.copy2(src_file, dst_file)
            
        for img in val_images:
            src_file = os.path.join(src_path, img)
            dst_file = os.path.join(dst_root, 'val', disease, img)
            shutil.copy2(src_file, dst_file)
            
        print(f"{disease}: {len(train_images)} ảnh train, {len(val_images)} ảnh validation")

# Data augmentation và chuẩn hóa
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Giảm kích thước ảnh
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Giảm kích thước ảnh
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Hàm huấn luyện cho một epoch
def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)  # Tối ưu hơn cho bộ nhớ
        
        # Sử dụng mixed precision để tiết kiệm bộ nhớ
        with autocast():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': (running_corrects.double()/total).item()})
        
        # Dọn bộ nhớ
        del outputs, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    
    return epoch_loss, epoch_acc

# Hàm đánh giá
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    
    return epoch_loss, epoch_acc

def main():
    # Cấu hình
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Đường dẫn dataset
    src_root = "Dữ liệu"
    dst_root = "dataset"
    
    # Chuẩn bị dataset
    print("Đang chuẩn bị dataset...")
    prepare_dataset(src_root, dst_root)
    
    # Tạo transforms
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(dst_root, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(dst_root, 'val'),
        transform=val_transform
    )
    
    # Create dataloaders với batch size nhỏ hơn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Giảm batch size
        shuffle=True,
        num_workers=0,  # Không dùng multiple workers trên Windows
        pin_memory=False  # Tắt pin_memory để tiết kiệm RAM
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16,  # Giảm batch size
        shuffle=False,
        num_workers=0,  # Không dùng multiple workers trên Windows
        pin_memory=False  # Tắt pin_memory để tiết kiệm RAM
    )
    
    # Khởi tạo model nhẹ hơn
    print("Đang khởi tạo model...")
    model = models.resnet18(pretrained=True)  # Dùng ResNet18 thay vì ResNet50
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)
    
    # Loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Thêm weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=5,  # Tăng patience
        min_lr=1e-6  # Thêm min learning rate
    )
    
    # Khởi tạo scaler cho mixed precision training
    scaler = GradScaler()
    
    # Huấn luyện
    print("Bắt đầu huấn luyện...")
    num_epochs = 20  # Giảm số epochs
    best_acc = 0.0
    
    try:
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train phase
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            
            # Validation phase
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            # Scheduler step
            scheduler.step(val_acc)
            
            # Lưu model tốt nhất
            if val_acc > best_acc:
                best_acc = val_acc
                # Lưu model dạng script để giảm kích thước file
                scripted_model = torch.jit.script(model)
                scripted_model.save('models/sugarcane_disease_model.pth')
                print(f'Model saved with accuracy: {best_acc:.4f}')
            
            # Dọn bộ nhớ sau mỗi epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nHuấn luyện bị dừng bởi người dùng!")
        # Lưu model checkpoint nếu bị dừng
        if best_acc > 0:
            torch.jit.script(model).save('models/sugarcane_disease_model_interrupted.pth')
    
    print("\nHuấn luyện hoàn tất!")
    print(f"Độ chính xác tốt nhất trên tập validation: {best_acc:.4f}")

if __name__ == "__main__":
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs('models', exist_ok=True)
    main()
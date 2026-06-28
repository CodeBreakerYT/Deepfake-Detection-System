import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import math

# Force PyTorch to use the local sample directory instead of C Drive cache
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TORCH_HOME'] = os.path.join(base_dir, "..", "sample")

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    filepath = os.path.join(dirpath, filename)
                    label = self._get_label_from_path(filepath)
                    if label is not None:
                        self.samples.append((filepath, label))

    def _get_label_from_path(self, filepath):
        import os
        filename = os.path.basename(filepath).lower()
        parent_dir = os.path.basename(os.path.dirname(filepath)).lower()
        path_lower = f"{parent_dir}/{filename}"
        if 'fake' in path_lower or 'manipulated' in path_lower:
            return 1
        elif 'real' in path_lower or 'original' in path_lower:
            return 0
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms

def train_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "datasets", "image")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Scanning dataset directory: {dataset_dir}")
    train_transforms = get_transforms()
    dataset = DeepfakeDataset(dataset_dir, transform=train_transforms)
    
    total_images = len(dataset)
    print(f"Found {total_images} valid images.")

    if total_images == 0:
        print("No images found! Cannot train model.")
        return

    # Calculate class weights for Imbalance
    fake_count = sum([1 for _, label in dataset.samples if label == 1])
    real_count = total_images - fake_count
    print(f"Dataset split -> Real: {real_count}, Fake: {fake_count}")

    weight_real = total_images / (2.0 * max(real_count, 1))
    weight_fake = total_images / (2.0 * max(fake_count, 1))
    class_weights = torch.FloatTensor([weight_real, weight_fake])
    print(f"Applied Class Weights: {class_weights}")

    # To ensure it doesn't take days, we will sample the dataset if it's too large for a demo run
    # For a full run, set num_samples to total_images
    MAX_SAMPLES = min(total_images, 50000) # Use at most 50k images for this robust run so it finishes in a few hours
    if MAX_SAMPLES < total_images:
        print(f"Subsampling {MAX_SAMPLES} images for faster training cycle.")
        indices = torch.randperm(len(dataset))[:MAX_SAMPLES]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    print(f"Using device: {device}")

    # Load ResNeXt50_32x4d (Robust Image Classifier)
    print("Loading ResNeXt50_32x4d with pretrained weights from local sample directory...")
    model = models.resnext50_32x4d(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: 0=Real, 1=Fake
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    epochs = 3 # 3 Epochs for a good balance of time/performance
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Starting Training (Robust ResNeXt50)...")
    best_loss = float('inf')
    model_path = os.path.join(models_dir, "IMG_MODEL.pth")
    checkpoint_path = os.path.join(models_dir, "IMG_MODEL_checkpoint.pth")
    
    start_epoch = 0
    start_step = 0
    running_loss = 0.0
    correct = 0
    total = 0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            best_loss = checkpoint['best_loss']
            running_loss = checkpoint.get('running_loss', 0.0)
            correct = checkpoint.get('correct', 0)
            total = checkpoint.get('total', 0)
            print(f"Resumed at epoch {start_epoch+1}, step {start_step}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    for epoch in range(start_epoch, epochs):
        model.train()
        
        if not (epoch == start_epoch and start_step > 0):
            running_loss = 0.0
            correct = 0
            total = 0

        for i, (inputs, labels) in enumerate(dataloader):
            if epoch == start_epoch and i < start_step:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # Save checkpoint every 100 steps
            if (i + 1) % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'step': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'running_loss': running_loss,
                    'correct': correct,
                    'total': total
                }, checkpoint_path)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        scheduler.step()

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved to {model_path} (Loss: {best_loss:.4f})")

        # Save checkpoint for the start of next epoch
        torch.save({
            'epoch': epoch + 1,
            'step': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'running_loss': 0.0,
            'correct': 0,
            'total': 0
        }, checkpoint_path)

if __name__ == "__main__":
    train_model()

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TORCH_HOME'] = os.path.join(base_dir, "..", "sample")

class VideoDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048):
        super(VideoDeepfakeModel, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        # Remove avgpool and fc layer
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, batch_first=True)
        self.dp = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        
        fmap = self.cnn(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        
        lstm_out, _ = self.lstm(x)
        out = torch.mean(lstm_out, dim=1)
        out = self.dp(self.fc(out))
        return out

class DeepfakeVideoDataset(Dataset):
    def __init__(self, root_dir, seq_length=15, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length
        self.samples = []

        valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}

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

    def extract_frames(self, filepath):
        cap = cv2.VideoCapture(filepath)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.seq_length:
            step = 1
        else:
            step = max(1, total_frames // self.seq_length)
            
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                if self.transform: img = self.transform(img)
                frames.append(img)
            count += 1
            if len(frames) == self.seq_length: break
        cap.release()
        
        # Pad if video was too short
        while len(frames) < self.seq_length:
            if len(frames) > 0:
                frames.append(frames[-1].clone())
            else:
                dummy = Image.new("RGB", (224, 224), (0,0,0))
                if self.transform: dummy = self.transform(dummy)
                frames.append(dummy)
                
        return torch.stack(frames)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            frames = self.extract_frames(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            dummy = Image.new("RGB", (224, 224), (0,0,0))
            if self.transform: dummy = self.transform(dummy)
            frames = torch.stack([dummy] * self.seq_length)
        return frames, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def train_model():
    dataset_dir = os.path.join(base_dir, "datasets", "video")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Scanning video dataset directory: {dataset_dir}")
    train_transforms = get_transforms()
    dataset = DeepfakeVideoDataset(dataset_dir, seq_length=15, transform=train_transforms)
    
    total_videos = len(dataset)
    print(f"Found {total_videos} valid videos.")

    if total_videos == 0:
        print("No videos found! Cannot train model.")
        return

    fake_count = sum([1 for _, label in dataset.samples if label == 1])
    real_count = total_videos - fake_count
    weight_real = total_videos / (2.0 * max(real_count, 1))
    weight_fake = total_videos / (2.0 * max(fake_count, 1))
    class_weights = torch.FloatTensor([weight_real, weight_fake])
    print(f"Dataset split -> Real: {real_count}, Fake: {fake_count}")

    MAX_SAMPLES = min(total_videos, 5000)
    if MAX_SAMPLES < total_videos:
        indices = torch.randperm(len(dataset))[:MAX_SAMPLES]
        dataset = torch.utils.data.Subset(dataset, indices)

    # We will create dataloader per epoch to correctly handle mid-epoch resuming without loading skipped data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_steps_per_epoch = (len(dataset) + 3) // 4

    print("Loading ResNeXt50+LSTM Video Model...")
    model = VideoDeepfakeModel().to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    
    epochs = 3
    checkpoint_path = os.path.join(models_dir, "VID_MODEL_checkpoint.pth")
    model_path = os.path.join(models_dir, "VID_MODEL.pth")
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    running_loss = 0.0
    correct = 0
    total = 0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            start_step = checkpoint.get('step', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            running_loss = checkpoint.get('running_loss', 0.0)
            correct = checkpoint.get('correct', 0)
            total = checkpoint.get('total', 0)
            print(f"Resumed at epoch {start_epoch+1}, step {start_step}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    print("Starting Video Training...")
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            
            # Create DataLoader for this epoch
            epoch_dataset = dataset
            if epoch == start_epoch and start_step > 0:
                # Skip the items that have already been processed to avoid slow frame extraction
                items_to_skip = start_step * 4
                if items_to_skip < len(dataset):
                    indices = torch.randperm(len(dataset)).tolist()
                    epoch_dataset = torch.utils.data.Subset(dataset, indices[items_to_skip:])
            
            dataloader = DataLoader(epoch_dataset, batch_size=4, shuffle=True, num_workers=0)
            
            if not (epoch == start_epoch and start_step > 0):
                running_loss = 0.0
                correct, total = 0, 0
            
            for i_rel, (inputs, labels) in enumerate(dataloader):
                i = i_rel + (start_step if epoch == start_epoch else 0)

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
                
                if (i + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps_per_epoch}], Loss: {loss.item():.4f}")

                # Save checkpoint periodically
                if (i + 1) % 50 == 0:
                    torch.save({
                        'epoch': epoch,
                        'step': i + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'running_loss': running_loss,
                        'correct': correct,
                        'total': total
                    }, checkpoint_path)

            epoch_loss = running_loss / (len(dataloader) + (start_step if epoch == start_epoch else 0))
            print(f"Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {epoch_loss:.4f}, Acc: {100*correct/total:.2f}%")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved to {model_path}")
                
            torch.save({
                'epoch': epoch + 1,
                'step': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'running_loss': 0.0,
                'correct': 0,
                'total': 0
            }, checkpoint_path)
            
    except KeyboardInterrupt:
        print("\n[PAUSE] Training interrupted by user! Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'step': i + 1 if 'i' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'running_loss': running_loss,
            'correct': correct,
            'total': total
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train_model()

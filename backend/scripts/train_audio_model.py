import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Numba/Librosa compatibility patch for NumPy 2.5
import builtins
np.__version__ = "2.0.0" 
if not hasattr(np, 'row_stack'):
    np.row_stack = np.vstack
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
if not hasattr(np, 'in1d'):
    np.in1d = np.isin
for t in ['complex', 'float', 'int', 'bool', 'object', 'str']:
    if not hasattr(np, t):
        setattr(np, t, eval(t) if t != 'str' else str)

import librosa

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TORCH_HOME'] = os.path.join(base_dir, "..", "sample")

class AudioSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioSpectrogramCNN, self).__init__()
        # Use ResNet18 adapted for single-channel (grayscale) spectrogram inputs
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.cnn(x)

class DeepfakeAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

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
        if 'fake' in path_lower or 'manipulated' in path_lower or 'synth' in path_lower:
            return 1
        elif 'real' in path_lower or 'original' in path_lower or 'human' in path_lower:
            return 0
        return None

    def __len__(self):
        return len(self.samples)

    def extract_mel_spectrogram(self, filepath):
        y, sr = librosa.load(filepath, sr=16000, duration=5.0)
        # Pad if audio is too short
        if len(y) < 16000 * 5:
            y = np.pad(y, (0, max(0, 16000 * 5 - len(y))), "constant")
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-255 for PIL Image
        mel_spec_db = mel_spec_db - mel_spec_db.min()
        mel_spec_db = mel_spec_db / (mel_spec_db.max() + 1e-8) * 255.0
        
        img = Image.fromarray(mel_spec_db.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            spectrogram = self.extract_mel_spectrogram(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            dummy = Image.new("L", (224, 224), 0)
            if self.transform: spectrogram = self.transform(dummy)
            
        return spectrogram, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Single channel normalization
    ])

def train_model():
    dataset_dir = os.path.join(base_dir, "datasets", "audio")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Scanning audio dataset directory: {dataset_dir}")
    train_transforms = get_transforms()
    dataset = DeepfakeAudioDataset(dataset_dir, transform=train_transforms)
    
    total_audios = len(dataset)
    print(f"Found {total_audios} valid audio files.")

    if total_audios == 0:
        print("No audio files found! Cannot train model.")
        return

    fake_count = sum([1 for _, label in dataset.samples if label == 1])
    real_count = total_audios - fake_count
    weight_real = total_audios / (2.0 * max(real_count, 1))
    weight_fake = total_audios / (2.0 * max(fake_count, 1))
    class_weights = torch.FloatTensor([weight_real, weight_fake])
    print(f"Dataset split -> Real: {real_count}, Fake: {fake_count}")

    MAX_SAMPLES = min(total_audios, 10000)
    if MAX_SAMPLES < total_audios:
        indices = torch.randperm(len(dataset))[:MAX_SAMPLES]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading AudioSpectrogramCNN (ResNet18-based)...")
    model = AudioSpectrogramCNN().to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    epochs = 5
    checkpoint_path = os.path.join(models_dir, "AUD_MODEL_checkpoint.pth")
    model_path = os.path.join(models_dir, "AUD_MODEL.pth")
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

    print("Starting Audio Training...")
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            
            if not (epoch == start_epoch and start_step > 0):
                running_loss = 0.0
                correct, total = 0, 0
            
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
                
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

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

            epoch_loss = running_loss / len(dataloader)
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

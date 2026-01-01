"""
Eko-style ResNet24 for Heart Murmur Detection
Based on: "Screening of heart murmur in adults via digital stethoscope"

Key aspects:
- Raw waveform input (no spectrograms)
- 8th-order Butterworth high-pass filter at 30 Hz
- Downsample to 2000 Hz
- ResNet24 architecture with 1D convolutions
- 3-class output: Murmur Present / Murmur Absent / Inadequate Signal (optional)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tqdm import tqdm

# Preprocessing Like Eko paper
def butter_highpass(cutoff, fs, order=8):
    """8th-order Butterworth high-pass filter at 30 Hz (Eko paper)"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=30, fs=2000, order=8):
    """Apply high-pass filter"""
    b, a = butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, data)

def load_and_preprocess_wav(file_path, target_fs=2000, max_length=30.0):
    """
    Load and preprocess WAV file following Eko methodology:
    1. Load audio
    2. Resample to 2000 Hz
    3. Apply 8th-order Butterworth high-pass at 30 Hz
    4. Normalize
    """
    # Load audio
    fs, data = wavfile.read(file_path)
    
    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    
    # Resample to target frequency
    if fs != target_fs:
        num_samples = int(len(data) * target_fs / fs)
        data = resample(data, num_samples)
    
    # Limit samples in case too much data
    max_samples = int(max_length * target_fs)
    if len(data) > max_samples:
        data = data[:max_samples]
    
    # Apply high-pass filter (30 Hz, 8th-order)
    data = highpass_filter(data, cutoff=30, fs=target_fs, order=8)
    
    # Normalize to [-1, 1]
    data = data / (np.max(np.abs(data)) + 1e-8)
    
    return data

# ----------------------------
# CirCor Dataset Loader
# ----------------------------
class CirCorMurmurDataset(Dataset):
    """
    Dataset for CirCor with murmur classification
    Can do either:
    - Binary: Present/Absent (murmur labels)
    - Binary: Normal/Abnormal (outcome labels)
    """
    def __init__(self, data_dir, label_type='murmur',
                 ausc_locations=['AV','PV','TV','MV','Phc'], 
                 target_fs=2000, max_length=30.0):
        """
        Args:
            data_dir: Path to CirCor training data
            label_type: 'murmur' (Present/Absent/Unknown) or 'outcome' (Normal/Abnormal)
            ausc_locations: Which locations to include
            target_fs: Target sampling frequency (2000 Hz)
            max_length: Max audio length in seconds
        """
        self.data_dir = Path(data_dir)
        self.label_type = label_type
        self.ausc_locations = ausc_locations
        self.target_fs = target_fs
        self.max_length = max_length
        self.files = []
        self.labels = []
        
        self._load_dataset()

    def _load_dataset(self):
        """Load all recordings with labels"""
        txt_files = list(self.data_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} subject files")
        
        for txt_file in txt_files:
            subject_id = txt_file.stem
            
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            header = lines[0].strip().split()
            num_recordings = int(header[1])
            
            # Extract labels from metadata
            outcome_label = None
            murmur_label = None
            
            for line in lines:
                if line.startswith('#Outcome:'):
                    outcome_label = line.strip().split(':')[-1].strip()
                elif line.startswith('#Murmur:'):
                    murmur_label = line.strip().split(':')[-1].strip()
            
            # Select label based on label_type
            if self.label_type == 'outcome':
                if outcome_label not in ['Normal', 'Abnormal']:
                    continue
                selected_label = outcome_label
            else:  # murmur
                if murmur_label not in ['Present', 'Absent']:
                    continue  # Skip 'Unknown'
                selected_label = murmur_label
            
            # Load recordings
            for i in range(1, num_recordings + 1):
                parts = lines[i].strip().split()
                location = parts[0]
                wav_file = parts[2]
                wav_path = self.data_dir / wav_file
                
                if location in self.ausc_locations and wav_path.exists():
                    self.files.append(str(wav_path))
                    self.labels.append(selected_label)
        
        print(f"Loaded {len(self.files)} recordings")
        
        # Encode labels
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
        
        # Print distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\nClass distribution ({self.label_type}):")
        for label_idx, count in zip(unique, counts):
            label_name = self.le.inverse_transform([label_idx])[0]
            print(f"  {label_name}: {count} recordings ({100*count/len(self.labels):.1f}%)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        
        # Load and preprocess
        data = load_and_preprocess_wav(wav_path, 
                                       target_fs=self.target_fs, 
                                       max_length=self.max_length)
        
        # Convert to tensor (1, L) - shape for 1D CNN
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return data, label

def collate_fn(batch):
    """Pad sequences to max length in batch"""
    data_list, label_list = zip(*batch)
    
    # Find max length
    max_length = max([d.shape[1] for d in data_list])
    
    # Padding
    padded_data = []
    for data in data_list:
        if data.shape[1] < max_length:
            padding = torch.zeros(1, max_length - data.shape[1])
            data = torch.cat([data, padding], dim=1)
        padded_data.append(data)
    
    return torch.stack(padded_data), torch.stack(label_list)

# ResNet24 Architecture (Eko paper)
class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D convolutions
    Structure: Conv1D -> BN -> ReLU -> Dropout -> Conv1D -> BN -> (+residual) -> ReLU -> MaxPool
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection (1x1 conv if channels change)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Residual connection
        out = self.relu(out)
        out = self.pool(out)
        
        return out

class ResNet24_1D(nn.Module):
    """
    ResNet24 for raw audio classification
    
    Architecture from Eko paper:
    - 34 layers total (convolution operations)
    - 4 stages with [16, 32, 64, 128] channels
    - 2 residual blocks per stage
    - Global average pooling
    - Fully connected output
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.2):
        super(ResNet24_1D, self).__init__()
        
        # Channel progression: 16 -> 32 -> 64 -> 128
        channels = [16, 32, 64, 128]
        
        # Build residual blocks
        layers = []
        input_ch = in_channels
        
        for ch in channels:
            # 2 residual blocks per stage
            for _ in range(2):
                layers.append(ResidualBlock1D(input_ch, ch, dropout=dropout))
                input_ch = ch
        
        self.res_blocks = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x shape: (batch, 1, length)
        out = self.res_blocks(x)
        out = self.global_pool(out).squeeze(-1)  # (batch, channels)
        out = self.fc(out)  # (batch, num_classes)
        return out  # Logits for CrossEntropyLoss

# Training Functions
def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100*correct/total:.2f}%'})
    
    return total_loss / total, correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return total_loss / total, correct / total, all_preds, all_labels

def compute_metrics(preds, labels):
    """Compute sensitivity and specificity (per Eko paper)"""
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds))
    
    # Sensitivity and Specificity (assuming binary: 0=absent, 1=present)
    if len(np.unique(labels)) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\nSensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda',
                save_path='best_eko_resnet24.pth'):
    """Complete training loop"""
    model.to(device)
    
    # ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 15
    
    print(f"\nTraining on device: {device}")
    print("="*70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-"*70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                           criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, 
                                                             criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Compute detailed metrics
        if epoch % 1 == 0:  # Every 5 epochs
            compute_metrics(val_preds, val_labels)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"Saved best model (acc: {val_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print(f"Training complete! Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {save_path}")
    
    return model

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Configuration matching Eko paper
    data_dir = "CirCor_DigiScope/training_data"
    batch_size = 128
    epochs = 50
    learning_rate = 1e-3
    target_fs = 2000  # Eko downsamples to 2000 Hz
    max_length = 30.0  # 30 seconds max
    label_type = 'murmur'  # 'murmur' or 'outcome'
    
    print("="*70)
    print("ResNet24 for Heart Murmur Detection")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sampling rate: {target_fs} Hz")
    print(f"  High-pass filter: 30 Hz (8th-order Butterworth)")
    print(f"  Max audio length: {max_length}s")
    print(f"  Label type: {label_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Check data directory
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory '{data_dir}' not found!")
        print("Please update the data_dir variable.")
        exit(1)
    
    # Load dataset
    print("\n" + "="*70)
    print("Loading CirCor Dataset...")
    print("="*70)
    
    dataset = CirCorMurmurDataset(
        data_dir,
        label_type=label_type,
        ausc_locations=['AV', 'PV', 'TV', 'MV', 'Phc'],
        target_fs=target_fs,
        max_length=max_length
    )
    
    # Train/val split (stratified)
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet24_1D(in_channels=1, num_classes=2, dropout=0.3)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Train
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        save_path='best_eko_resnet24.pth'
    )
    
    print("\nTraining completed successfully!")
    print("="*70)
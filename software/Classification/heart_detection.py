"""
Eko-style ResNet for Heart Murmur Detection
Finds best threshold using youden thresholding

Based on: "Screening of heart murmur in adults via digital stethoscope"
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


## Preprocessing

def butter_highpass(cutoff, fs, order=8):
    """8th-order Butterworth high-pass filter at 30 Hz"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def highpass_filter(data, cutoff=30, fs=2000, order=8):
    """Apply 8th-order Butterworth high-pass filter at 30 Hz"""
    b, a = butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def load_and_preprocess_wav(file_path, target_fs=2000, max_length=30.0):
    """Load and preprocess WAV file following Eko methodology"""
    fs, data = wavfile.read(file_path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if len(data.shape) > 1:
        data = data[:, 0]

    # High-pass filter at original sample rate
    if fs >= 60:
        data = highpass_filter(data, cutoff=30, fs=fs, order=8)

    # Downsample to 2000 Hz
    if fs != target_fs:
        num_samples = int(len(data) * target_fs / fs)
        data = resample(data, num_samples)

    # Limit to max length
    max_samples = int(max_length * target_fs)
    if len(data) > max_samples:
        data = data[:max_samples]

    # Normalize
    max_val = np.max(np.abs(data))
    if max_val > 1e-8:
        data = data / max_val

    return data


## Data Augmentation

class AudioAugmentation:
    """Audio augmentations for minority class (murmur class)"""

    @staticmethod
    def time_shift(data, shift_max=0.2):
        shift = int(len(data) * np.random.uniform(-shift_max, shift_max))
        return np.roll(data, shift)

    @staticmethod
    def add_noise(data, noise_level=0.005):
        noise = np.random.randn(len(data)) * noise_level
        return data + noise

    @staticmethod
    def time_stretch(data, rate_range=(0.9, 1.1)):
        rate = np.random.uniform(*rate_range)
        new_len = int(len(data) / rate)
        return resample(data, new_len)

    @staticmethod
    def amplitude_scale(data, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(*scale_range)
        return data * scale


## Dataset

class CirCorMurmurDataset(Dataset):
     """
    Dataset for the CirCor heart sound dataset with
    data augmentation and class imbalance handling.
    """

    def __init__(self, data_dir, label_type='murmur_binary',
                 ausc_locations=['AV', 'PV', 'TV', 'MV', 'Phc'],
                 target_fs=2000, max_length=30.0,
                 augment=False, augment_minority_only=True):
      
        # Root directory containing .txt metadata and .wav files
        self.data_dir = Path(data_dir)

        # Type of label to use (currently binary murmur classification)
        self.label_type = label_type

        # Auscultation locations to include
        self.ausc_locations = ausc_locations

        # Target sampling rate (Hz) for resampling audio
        self.target_fs = target_fs

        # Maximum signal duration (in seconds)
        self.max_length = max_length

        # Data augmentation flags
        self.augment = augment
        self.augment_minority_only = augment_minority_only

        # Lists to store file paths and corresponding labels
        self.files = []
        self.labels = []

        # Load metadata and populate file/label lists
        self._load_dataset()
        self.augmenter = AudioAugmentation()

    # Load dataset metadata and extract file paths and murmur labels
    def _load_dataset(self):
        txt_files = list(self.data_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} subject files")

        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            # First line contains subject metadata
            header = lines[0].strip().split()
            num_recordings = int(header[1])

            # Find murmur label from header comments
            murmur_label = None
            for line in lines:
                if line.startswith('#Murmur:'):
                    murmur_label = line.strip().split(':')[-1].strip()

            # Skip individuals without clear murmur labels
            if murmur_label not in ['Present', 'Absent']:
                continue

            # Binary encoding: 1 = Murmur present, 0 = No murmur
            selected_label = 1 if murmur_label == 'Present' else 0

            # Iterate over individual recordings for this subject
            for i in range(1, num_recordings + 1):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) < 3:
                    continue
                location = parts[0]
                wav_file = parts[2]
                wav_path = self.data_dir / wav_file

                # Only include desired auscultation locations
                if location in self.ausc_locations and wav_path.exists():
                    self.files.append(str(wav_path))
                    self.labels.append(selected_label)

        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.files)} recordings")

        # Find class distribution for imbalance handling
        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))
        print(f"\nClass distribution:")
        print(f"  No Murmur (0): {self.class_counts.get(0, 0)} ({100*self.class_counts.get(0, 0)/len(self.labels):.1f}%)")
        print(f"  Murmur (1): {self.class_counts.get(1, 0)} ({100*self.class_counts.get(1, 0)/len(self.labels):.1f}%)")

        # Find minority class for imbalance augmentation
        self.minority_class = 1 if self.class_counts.get(1, 0) < self.class_counts.get(0, 0) else 0

    # Dataset length
    def __len__(self):
        return len(self.files)
    
    # Get a single sample (audio and label)
    def __getitem__(self, idx):
        wav_path = self.files[idx]
        label = self.labels[idx]

        # Load waveform, resample, and trim/pad to max length
        data = load_and_preprocess_wav(wav_path, self.target_fs, self.max_length)

        # Data augmentation
        if self.augment:
            should_augment = (not self.augment_minority_only) or (label == self.minority_class)
            if should_augment and np.random.random() > 0.5:
                aug_choice = np.random.randint(4)
                if aug_choice == 0:
                    data = self.augmenter.time_shift(data)
                elif aug_choice == 1:
                    data = self.augmenter.add_noise(data)
                elif aug_choice == 2:
                    data = self.augmenter.amplitude_scale(data)
                else:
                    data = self.augmenter.time_stretch(data)

                max_val = np.max(np.abs(data))
                if max_val > 1e-8:
                    data = data / max_val

        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return data, label

    # Calculate class weights for weighted sampling
    # to correcting class imbalance during training
    def get_sample_weights(self):
        class_weights = {
            0: 1.0 / self.class_counts[0],
            1: 1.0 / self.class_counts[1]
        }
        weights = [class_weights[label] for label in self.labels]
        return torch.DoubleTensor(weights)

# Custom collate function to pads all samples in a batch to the maximum length
def collate_fn(batch):
    data_list, label_list = zip(*batch)

    # Determine max temporal length in the batch
    max_length = max([d.shape[1] for d in data_list])
    
    padded_data = []
    for data in data_list:
        if data.shape[1] < max_length:
            padding = torch.zeros(1, max_length - data.shape[1])
            data = torch.cat([data, padding], dim=1)
        padded_data.append(data)

    return torch.stack(padded_data), torch.stack(label_list)


## Model

class EkoResNet34(nn.Module):
    """34-layer ResNet matching Eko paper"""

    def __init__(self, in_channels=1, num_classes=2, dropout=0.2, kernel_size=7):
        super(EkoResNet34, self).__init__()

        # Initial feature extraction block:
        # 1. 1d convolution
        # 2. BatchNorm + ReLU for stable training
        # 3. Dropout for regularization
        # 4. Downsampling via max pooling
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
            nn.MaxPool1d(4)
        )

        # Five sequential stages of convolutional layers
        # Each stage increases channel depth and reduces temporal resolution
        # pools_at specifies which layers apply temporal downsampling
        self.stage1 = self._make_stage(16, 32, 6, dropout, pools_at=[0, 3])
        self.stage2 = self._make_stage(32, 64, 7, dropout, pools_at=[0, 4])
        self.stage3 = self._make_stage(64, 128, 7, dropout, pools_at=[0, 4])
        self.stage4 = self._make_stage(128, 256, 7, dropout, pools_at=[0, 4])
        self.stage5 = self._make_stage(256, 256, 6, dropout, pools_at=[0, 3])
        
        # Global average pooling collapses the time dimension to length 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final fully connected classification layer
        self.fc = nn.Linear(256, num_classes)

    def _make_stage(self, in_ch, out_ch, num_layers, dropout, pools_at):
        """
        Creates a stage consisting of multiple convolutional layers.
        Some layers optionally apply max pooling for downsampling.
        """

        layers = []
        current_ch = in_ch
        for i in range(num_layers):
            # Apply pooling only at specified layer indices
            use_pool = i in pools_at
            layers.append(self._make_layer(current_ch, out_ch, dropout, use_pool))
            current_ch = out_ch
        return nn.Sequential(*layers)

    def _make_layer(self, in_ch, out_ch, dropout, use_pool):
        """
        Single convolutional block:
        Conv -> ReLU -> BatchNorm -> Dropout (+ optional MaxPool)
        """
        components = [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(dropout),
        ]
        if use_pool:
            components.append(nn.MaxPool1d(2))
        return nn.Sequential(*components)

    def forward(self, x):
        """
        Forward pass:
        Input shape: (batch, channels, time)
        """

        out = self.initial_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.global_pool(out).squeeze(-1)

        # Final classification
        out = self.fc(out)
        return out



# THRESHOLD TUNING

class ThresholdTuner:
    """
    Find optimal decision threshold for balancing sensitivity and specificity.
    """

    def __init__(self, labels, probs):
        """
        Args:
            labels: Ground truth labels (0 = No Murmur, 1 = Murmur)
            probs: Predicted probabilities for Murmur class
        """
        self.labels = np.array(labels)
        self.probs = np.array(probs)
        self.fpr, self.tpr, self.thresholds = roc_curve(self.labels, self.probs)
        self.auc = roc_auc_score(self.labels, self.probs)

    def get_metrics_at_threshold(self, threshold):
         """
        Compute performance metrics for a given threshold

        Args:
            threshold: Probability cutoff for positive classification

        Returns:
            Dictionary containing sensitivity, specificity, precision,
            NPV, F1 score, and confusion matrix components.
        """
        # Convert probabilities to binary predictions
        preds = (self.probs >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(self.labels, preds)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

            return {
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'npv': npv,
                'f1': f1,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        return None

    def find_youden_threshold(self):
        """
        Find threshold that maximizes Youden's J statistic:
        J = sensitivity + specificity - 1
        """
        youden_j = self.tpr - self.fpr
        optimal_idx = np.argmax(youden_j)
        return self.thresholds[optimal_idx]

    # Not used anymore
    def find_sensitivity_threshold(self, target_sensitivity=0.85):
        """
        Find threshold that achieves at least the target sensitivity
        while maximizing specificity.
        """
        idx_meeting_target = np.where(self.tpr >= target_sensitivity)[0]
        if len(idx_meeting_target) > 0:
            # Among those meeting sensitivity target, find best specificity
            best_idx = idx_meeting_target[np.argmin(self.fpr[idx_meeting_target])]
            return self.thresholds[best_idx]
        return 0.5

    # Not used anymore
    def find_specificity_threshold(self, target_specificity=0.85):
        """Find threshold that achieves target specificity with best sensitivity"""
        specificities = 1 - self.fpr
        idx_meeting_target = np.where(specificities >= target_specificity)[0]
        if len(idx_meeting_target) > 0:
            # Among those meeting specificity target, find best sensitivity
            best_idx = idx_meeting_target[np.argmax(self.tpr[idx_meeting_target])]
            return self.thresholds[best_idx]
        return 0.5

    # Not used anymore
    def find_eer_threshold(self):
        """Find threshold where sensitivity ≈ specificity (Equal Error Rate)"""
        eer_idx = np.argmin(np.abs(self.tpr - (1 - self.fpr)))
        return self.thresholds[eer_idx]

    # Not used anymore
    def find_f1_threshold(self):
        """Find threshold that maximizes F1 score"""
        best_f1 = 0
        best_thresh = 0.5

        for thresh in np.arange(0.1, 0.9, 0.01):
            metrics = self.get_metrics_at_threshold(thresh)
            if metrics and metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_thresh = thresh

        return best_thresh

    # Run full analysis and return metrics
    def analyze(self, target_sensitivity=0.85, target_specificity=0.85):
        """Run full threshold analysis and return results"""

        results = {
            'auc': self.auc,
            'thresholds': {},
            'metrics': {}
        }

        # Find different optimal thresholds
        results['thresholds']['youden'] = self.find_youden_threshold()
        results['thresholds']['target_sensitivity'] = self.find_sensitivity_threshold(target_sensitivity)
        results['thresholds']['target_specificity'] = self.find_specificity_threshold(target_specificity)
        results['thresholds']['eer'] = self.find_eer_threshold()
        results['thresholds']['f1'] = self.find_f1_threshold()
        results['thresholds']['default'] = 0.5

        # Get metrics at each threshold
        for name, thresh in results['thresholds'].items():
            results['metrics'][name] = self.get_metrics_at_threshold(thresh)

        return results

    def print_analysis(self, target_sensitivity=0.85, target_specificity=0.85):
        """Print detailed threshold analysis"""
        results = self.analyze(target_sensitivity, target_specificity)

        print("\n" + "=" * 70)
        print("THRESHOLD ANALYSIS")
        print("=" * 70)
        print(f"\nAUC-ROC: {results['auc']:.4f}")

        print("\n" + "-" * 70)
        print(f"{'Method':<25} {'Threshold':<12} {'Sensitivity':<12} {'Specificity':<12} {'F1':<8}")
        print("-" * 70)

        for name in ['default', 'youden', 'eer', 'f1', 'target_sensitivity', 'target_specificity']:
            thresh = results['thresholds'][name]
            m = results['metrics'][name]
            if m:
                print(f"{name:<25} {thresh:<12.4f} {m['sensitivity']:<12.4f} {m['specificity']:<12.4f} {m['f1']:<8.4f}")

        print("-" * 70)

        # Recommendation
        youden_metrics = results['metrics']['youden']
        print(f"\n✓ RECOMMENDED (Youden's J - Best Balance):")
        print(f"  Threshold: {results['thresholds']['youden']:.4f}")
        print(f"  Sensitivity: {youden_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {youden_metrics['specificity']:.4f}")

        return results

    def plot_analysis(self, save_path='threshold_analysis.png'):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. ROC Curve
        ax1 = axes[0, 0]
        ax1.plot(self.fpr, self.tpr, 'b-', linewidth=2, label=f'ROC (AUC = {self.auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)

        # Mark key thresholds
        youden_thresh = self.find_youden_threshold()
        eer_thresh = self.find_eer_threshold()

        youden_idx = np.argmin(np.abs(self.thresholds - youden_thresh))
        eer_idx = np.argmin(np.abs(self.thresholds - eer_thresh))

        ax1.scatter([self.fpr[youden_idx]], [self.tpr[youden_idx]],
                   color='red', s=100, zorder=5, label=f'Youden (t={youden_thresh:.2f})')
        ax1.scatter([self.fpr[eer_idx]], [self.tpr[eer_idx]],
                   color='green', s=100, zorder=5, label=f'EER (t={eer_thresh:.2f})')

        ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # 2. Sensitivity & Specificity vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(self.thresholds, self.tpr, 'b-', linewidth=2, label='Sensitivity')
        ax2.plot(self.thresholds, 1 - self.fpr, 'r-', linewidth=2, label='Specificity')
        ax2.axvline(x=youden_thresh, color='green', linestyle='--', alpha=0.7, label=f'Youden ({youden_thresh:.2f})')
        ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Default (0.5)')
        ax2.set_xlabel('Threshold', fontsize=11)
        ax2.set_ylabel('Rate', fontsize=11)
        ax2.set_title('Sensitivity & Specificity vs Threshold', fontsize=12, fontweight='bold')
        ax2.legend(loc='center right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])

        # 3. Precision-Recall Curve
        ax3 = axes[1, 0]
        precision, recall, pr_thresholds = precision_recall_curve(self.labels, self.probs)
        ap = average_precision_score(self.labels, self.probs)
        ax3.plot(recall, precision, 'g-', linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
        ax3.set_xlabel('Recall (Sensitivity)', fontsize=11)
        ax3.set_ylabel('Precision', fontsize=11)
        ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower left')
        ax3.grid(True, alpha=0.3)

        # 4. Metrics vs Threshold
        ax4 = axes[1, 1]
        thresholds_range = np.arange(0.05, 0.95, 0.02)
        sensitivities = []
        specificities = []
        f1_scores = []

        for t in thresholds_range:
            m = self.get_metrics_at_threshold(t)
            if m:
                sensitivities.append(m['sensitivity'])
                specificities.append(m['specificity'])
                f1_scores.append(m['f1'])

        ax4.plot(thresholds_range, sensitivities, 'b-', linewidth=2, label='Sensitivity')
        ax4.plot(thresholds_range, specificities, 'r-', linewidth=2, label='Specificity')
        ax4.plot(thresholds_range, f1_scores, 'g-', linewidth=2, label='F1 Score')
        ax4.axvline(x=youden_thresh, color='purple', linestyle='--', alpha=0.7, label=f'Optimal ({youden_thresh:.2f})')
        ax4.set_xlabel('Threshold', fontsize=11)
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_title('All Metrics vs Threshold', fontsize=12, fontweight='bold')
        ax4.legend(loc='center right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {save_path}")


# Training

def compute_class_weights(labels, weight_ratio=None):
    """
    Compute class weights for handling class imbalance

    Args:
        labels: Array of labels
        weight_ratio: If provided, use manual ratio (e.g., 1.5 means murmur weighted 1.5x)
                     If None, use inverse frequency
    """
    unique, counts = np.unique(labels, return_counts=True)

    if weight_ratio is not None:
        # Manual weight ratio
        return {0: 1.0, 1: weight_ratio}
    else:
        # Inverse-frequency weighting to balance rare classes
        total = len(labels)
        weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
        return weights


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Run one training epoch

    Returns:
        Average loss and accuracy over the epoch.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    # Progress bar for training batches
    pbar = tqdm(train_loader, desc="Training")

    for x, y in pbar:
       # Move batch to GPU/CPU
        x, y = x.to(device), y.to(device)

        # Compute forward and backward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate statistics
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return total_loss / total, correct / total


def validate(model, val_loader, criterion, device):
     """
    Run validation loop

    Returns:
        Average loss, accuracy, predictions, labels, and murmur probabilities.
    """

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Store outputs for metrics and threshold tuning
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            loss = criterion(logits, y)

            # Calculate statistics
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def compute_metrics_at_threshold(labels, probs, threshold=0.5):
    """Compute and print metrics at a specific threshold"""
    preds = (np.array(probs) >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    print(f"\n--- Metrics at threshold = {threshold:.4f} ---")
    print(classification_report(labels, preds, target_names=['No Murmur', 'Murmur']))

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"Confusion Matrix:")
        print(f"  TN={tn:4d}  FP={fp:4d}")
        print(f"  FN={fn:4d}  TP={tp:4d}")
        print(f"\nSENSITIVITY: {sensitivity:.4f}")
        print(f"SPECIFICITY: {specificity:.4f}")

        return sensitivity, specificity
    return None, None


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda',
                save_path='best_model.pth', class_weights=None):
      """
    Full training loop with:
      - Class weighting
      - Learning-rate scheduling
      - Early stopping
      - Best-model checkpointing
    """

    model.to(device)

    # Loss function
    if class_weights:
        print(f"Using class weights: {class_weights}")
        weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_auc = 0.0
    patience_counter = 0
    max_patience = 15

    best_probs = None
    best_labels = None

    print(f"\nTraining on device: {device}")
    print("=" * 70)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)

        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # Validation
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Compute AUC
        try:
            auc = roc_auc_score(val_labels, val_probs)
            print(f"Val AUC: {auc:.4f}")
        except:
            auc = 0.0

        # Calculate metrics at default threshold
        compute_metrics_at_threshold(val_labels, val_probs, threshold=0.5)

        scheduler.step(auc)

        # Save best model based on AUC
        if auc > best_auc:
            best_auc = auc
            best_probs = val_probs.copy() if isinstance(val_probs, np.ndarray) else list(val_probs)
            best_labels = val_labels.copy() if isinstance(val_labels, np.ndarray) else list(val_labels)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': auc,
                'val_acc': val_acc,
            }, save_path)
            print(f"✓ Saved best model (AUC: {auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best AUC: {best_auc:.4f}")

    # Return best probs and labels for threshold tuning
    return model, best_labels, best_probs


## Murmur Detection Wrapper

class MurmurDetector:
    """
    Inference that applies a trained model with a tuned threshold
    """

    def __init__(self, model, threshold=0.5, device='cuda'):
        self.model = model
        self.model.eval()
        self.threshold = threshold
        self.device = device

    def predict(self, audio_path):
        """
        Predict murmur presence for a single audio file

        Returns:
            dict with prediction, probability, and threshold used
        """
        # Load and preprocess waveform
        data = load_and_preprocess_wav(audio_path, target_fs=2000, max_length=30.0)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        data = data.to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(data)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        murmur_prob = probs[1]
        prediction = "Murmur" if murmur_prob >= self.threshold else "No Murmur"

        return {
            'prediction': prediction,
            'murmur_probability': float(murmur_prob),
            'no_murmur_probability': float(probs[0]),
            'threshold': self.threshold,
            'confidence': float(max(probs))
        }

    def predict_batch(self, audio_paths):
        """Predict for multiple audio files"""
        results = []
        for path in audio_paths:
            results.append(self.predict(path))
        return results

#Main

if __name__ == "__main__":
    # Configuration
    data_dir = "CirCor_DigiScope/training_data"
    batch_size = 64
    epochs = 50
    learning_rate = 1e-3
    target_fs = 2000
    max_length = 30.0

    # Class imbalance settings
    USE_WEIGHTED_SAMPLING = True
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHT_RATIO = 1.5  # Murmur weighted 1.5x (can tune this)
    USE_AUGMENTATION = True

    print("=" * 70)
    print("EKO RESNET34 WITH THRESHOLD TUNING")
    print("=" * 70)

    if not os.path.exists(data_dir):
        print(f"\nError: Data directory '{data_dir}' not found!")
        exit(1)

    # Load dataset
    print("\nLoading dataset...")
    dataset = CirCorMurmurDataset(
        data_dir,
        label_type='murmur_binary',
        ausc_locations=['AV', 'PV', 'TV', 'MV', 'Phc'],
        target_fs=target_fs,
        max_length=max_length,
        augment=USE_AUGMENTATION,
        augment_minority_only=True
    )

    # Train/validation split
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)

    # Validation without augmentation
    val_dataset_no_aug = CirCorMurmurDataset(
        data_dir,
        label_type='murmur_binary',
        ausc_locations=['AV', 'PV', 'TV', 'MV', 'Phc'],
        target_fs=target_fs,
        max_length=max_length,
        augment=False
    )
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_idx)

    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Class weights
    train_labels = dataset.labels[train_idx]
    class_weights = compute_class_weights(train_labels, weight_ratio=CLASS_WEIGHT_RATIO) if USE_CLASS_WEIGHTS else None
    print(f"Class weights: {class_weights}")

    # Weighted sampling
    if USE_WEIGHTED_SAMPLING:
        sample_weights = dataset.get_sample_weights()[train_idx]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EkoResNet34(in_channels=1, num_classes=2, dropout=0.3)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model, val_labels, val_probs = train_model(
        model, train_loader, val_loader,
        epochs=epochs, lr=learning_rate, device=device,
        save_path='best_eko_model.pth',
        class_weights=class_weights
    )

    # Threshold Tuning

    print("\n" + "=" * 70)
    print("THRESHOLD TUNING")
    print("=" * 70)

    # Create threshold tuner
    tuner = ThresholdTuner(val_labels, val_probs)

    # Run analysis
    results = tuner.print_analysis(target_sensitivity=0.85, target_specificity=0.85)

    # Generate plots
    tuner.plot_analysis(save_path='threshold_analysis.png')

    # Get optimal threshold
    optimal_threshold = results['thresholds']['youden']

    print("\n" + "=" * 70)
    print("FINAL RESULTS WITH OPTIMAL THRESHOLD")
    print("=" * 70)

    compute_metrics_at_threshold(val_labels, val_probs, threshold=optimal_threshold)

    # Save the optimal threshold with the model
    checkpoint = torch.load('best_eko_model.pth')
    checkpoint['optimal_threshold'] = optimal_threshold
    checkpoint['threshold_results'] = results
    torch.save(checkpoint, 'best_eko_model.pth')

    print(f"\n✓ Model saved with optimal threshold: {optimal_threshold:.4f}")

    # ==============================================================================
    # Example on how to load trained model

    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print(f"""
# Load model and create detector
checkpoint = torch.load('best_eko_model.pth')
model = EkoResNet34(in_channels=1, num_classes=2, dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Use optimal threshold from training
optimal_threshold = checkpoint['optimal_threshold']  # {optimal_threshold:.4f}

# Create detector
detector = MurmurDetector(model, threshold=optimal_threshold, device=device)

# Predict on new audio
result = detector.predict('path/to/audio.wav')
print(result)
# {{'prediction': 'Murmur', 'murmur_probability': 0.73, 'threshold': {optimal_threshold:.4f}}}
""")

    print("\nDone!")

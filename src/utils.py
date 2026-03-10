import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import librosa



def load_data(config, task):
    """Load pre-processed data for task (sound or noise).
    
    Args:
        config: Dict from YAML.
        task: 'sound' or 'noise'.
    
    Returns:
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    """
    data_dir = config['paths']['data_dir']
    X_train = np.load(os.path.join(data_dir, 'Train_feature/X_train.npy')).astype(np.float32)
    Y_train = np.load(os.path.join(data_dir, f'Train_feature/Y_train_{task}.npy'))
    
    X_val = np.load(os.path.join(data_dir, 'Val_feature/X_val.npy')).astype(np.float32)
    Y_val = np.load(os.path.join(data_dir, f'Val_feature/Y_val_{task}.npy'))
    
    X_test = np.load(os.path.join(data_dir, 'Test_feature/X_test.npy')).astype(np.float32)
    Y_test = np.load(os.path.join(data_dir, f'Test_feature/Y_test_{task}.npy'))
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def plot_cm(y_true, y_pred, save_path, title):
    """Plot and save confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save PNG.
        title: Plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_roc(y_true, y_prob, num_classes, save_path):
    """Plot and save ROC curves.
    
    Args:
        y_true: True labels.
        y_prob: Probability predictions.
        num_classes: Number of classes.
        save_path: Path to save PNG.
    """
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def write_report(results_path, seed, best_epoch, best_val_acc, report, cm_path, roc_path):
    """Write classification report to TXT.
    
    Args:
        results_path: Path to TXT file.
        seed: Current seed.
        best_epoch: Early stopping epoch.
        best_val_acc: Best val accuracy.
        report: classification_report string.
        cm_path: CM plot path.
        roc_path: ROC plot path.
    """
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Seed: {seed} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Early Stopping at epoch: {best_epoch} (Best val_acc = {best_val_acc:.4f})\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nCM Plot: {cm_path}\n")
        f.write(f"ROC Plot: {roc_path}\n")

def normalize_db(data, mean, std):
    """DB normalization from notebook."""
    return np.expand_dims((data - mean) / (std + 1e-6), axis=-1)

def load_phase2_data(config):
    """Load data for phase 2 (same as phase 1 but normalized).
    
    Returns:
        train_data, val_data, test_data (each: X_spec, Y_bio, Y_noise)
    """
    data_dir = config['paths']['data_dir']
    X_train_spec = np.load(os.path.join(data_dir, 'Train_feature/X_train_spec.npy')).astype(np.float32)
    Y_bio_train = np.load(os.path.join(data_dir, 'Train_feature/X_train_bio.npy'))  # Note: notebook uses X_train_bio for Y_bio
    Y_noise_train = np.load(os.path.join(data_dir, 'Train_feature/X_train_noise.npy'))
    
    X_val_spec = np.load(os.path.join(data_dir, 'Val_feature/X_val_spec.npy')).astype(np.float32)
    Y_bio_val = np.load(os.path.join(data_dir, 'Val_feature/X_val_bio.npy'))
    Y_noise_val = np.load(os.path.join(data_dir, 'Val_feature/X_val_noise.npy'))
    
    X_test_spec = np.load(os.path.join(data_dir, 'Test_feature/X_test_spec.npy')).astype(np.float32)
    Y_bio_test = np.load(os.path.join(data_dir, 'Test_feature/X_test_bio.npy'))
    Y_noise_test = np.load(os.path.join(data_dir, 'Test_feature/X_test_noise.npy'))
    
    # Normalize using train stats
    train_mean = np.mean(X_train_spec)
    train_std = np.std(X_train_spec)
    
    X_train_spec = normalize_db(X_train_spec, train_mean, train_std)
    X_val_spec = normalize_db(X_val_spec, train_mean, train_std)
    X_test_spec = normalize_db(X_test_spec, train_mean, train_std)
    
    return (X_train_spec, Y_bio_train, Y_noise_train), \
           (X_val_spec, Y_bio_val, Y_noise_val), \
           (X_test_spec, Y_bio_test, Y_noise_test)

# Existing plot_cm, plot_roc, write_report can be reused/adapted for phase2 (add noise CM/ROC if needed)
def save_roc_csv(y_true, y_score, save_path):
    """Save ROC coords to CSV (voice only, as in notebook)."""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]
    roc_data = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        for fp, tp in zip(fpr, tpr):
            roc_data.append({'class': i, 'fpr': fp, 'tpr': tp, 'auc': roc_auc})
    pd.DataFrame(roc_data).to_csv(save_path, index=False)

def write_phase2_report(results_path, seed, best_epoch, best_val_acc, 
                        voice_report, noise_report, cm_voice_path, cm_noise_path, 
                        roc_plot_path, roc_csv_path):
    """Adapted write_report for phase2 (voice + noise)."""
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*90}\n")
        f.write(f"Experiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Loss Weights: voice_type: 5.0, noise_type: 1.0\n")
        f.write(f"Early Stopping at epoch: {best_epoch} (Best val_voice_acc = {best_val_acc:.4f})\n")
        f.write(f"{'='*90}\n\n")
        
        f.write("【VoiceType Classification Report】\n")
        f.write(voice_report)
        f.write(f"\nConfusion Matrix: {cm_voice_path}\n")
        f.write(f"ROC Plot: {roc_plot_path}\n")
        f.write(f"ROC CSV: {roc_csv_path}\n\n")
        
        f.write("【NoiseType Classification Report】\n")
        f.write(noise_report)
        


def load_and_segment_real_data(config):
    """Load and segment real PAM .wav files into 0.5s segments with 50% overlap.
    
    Args:
        config: Dict from YAML.
    
    Returns:
        X_real: np.array of segments (num_segments, samples)
        y_real: np.array of labels (0: WhiteBeaked, 1: WhiteSided, 2: AtlanticSpotted)
    """
    real_dir = config['real_test']['real_data_dir']
    sr = config['real_test']['sr']
    segment_length = config['real_test']['segment_length']
    segment_samples = int(segment_length * sr)
    
    audio_files = {
        0: 'WhiteBeaked.wav',  # Label 0
        1: 'WhiteSided.wav',   # Label 1
        2: 'AtlanticSpotted.wav'  # Label 2 (note: filename is 6102500S.wav in notebook, rename if needed)
    }
    
    X_real = []
    y_real = []
    
    for label, filename in audio_files.items():
        file_path = os.path.join(real_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {filename} in {real_dir}")
        
        y_audio, _ = librosa.load(file_path, sr=sr)
        for start in range(0, len(y_audio) - segment_samples + 1, segment_samples // 2):  # 50% overlap
            seg = y_audio[start:start + segment_samples]
            if len(seg) == segment_samples:
                X_real.append(seg)
                y_real.append(label)
    
    return np.array(X_real), np.array(y_real)

def generate_stft_spectrograms(segments, sr=60600):
    """Generate STFT spectrograms from segments (log-scaled, resized to 128x128).
    
    Args:
        segments: np.array of audio segments.
        sr: Sampling rate.
    
    Returns:
        specs: np.array (num_segments, 128, 128, 1)
    """
    specs = []
    for seg in segments:
        stft = librosa.stft(seg, n_fft=512, hop_length=256, window='hann')
        log_spec = librosa.amplitude_to_db(np.abs(stft))
        # Resize to 128x128 (use tf.image.resize if needed, but here approximate with slicing/averaging if exact)
        # For simplicity, assume log_spec is cropped/padded to 128x128; implement resize if shape differs
        log_spec_resized = log_spec[:128, :128]  # Placeholder; use scipy.misc.imresize or tf for actual resize
        specs.append(np.expand_dims(log_spec_resized, axis=-1))
    specs = np.array(specs)
    
    # Normalize (Z-score using mean/std from train if available; here global)
    mean = np.mean(specs)
    std = np.std(specs)
    specs = (specs - mean) / (std + 1e-6)
    return specs
"""
Voice Speaker Classifier - PyTorch 2-Layer Neural Network
==========================================================
Classifies audio samples as:
  Gabriel = 0
  Raiz    = 1

Pipeline:
  1. Trim silence/noise from raw .m4a files
  2. Segment into 5-second clips
  3. Convert each clip to a 128x128 Mel spectrogram (flattened to 16384 features)
  4. Build labeled dataset:
       - TRAIN: 100% of gabriel_samples.m4a + raiz_samples.m4a (all available clips)
       - TEST:  gabriel_test.m4a + raiz_test.m4a (balanced - capped to smaller class)
       - Global normalization: mean & std fitted on training set, applied to both
  5. Train a PyTorch 2-layer neural network (sigmoid hidden activation)
  6. Evaluate: ROC Curve, AUC, Confusion Matrix, Attribute Heatmap
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import librosa
import librosa.effects
import librosa.feature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydub import AudioSegment
from project_paths import (
    GABRIEL_TRAIN_FILE as PROJECT_GABRIEL_TRAIN_FILE,
    RAIZ_TRAIN_FILE as PROJECT_RAIZ_TRAIN_FILE,
    GABRIEL_TEST_FILE as PROJECT_GABRIEL_TEST_FILE,
    RAIZ_TEST_FILE as PROJECT_RAIZ_TEST_FILE,
    RESULTS_DIR,
    ensure_base_dirs,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SAMPLE_RATE = 22050
CLIP_DURATION = 5
SAMPLES_PER_CLIP = SAMPLE_RATE * CLIP_DURATION  # 110250
N_MELS = 128
IMG_SIZE = 128
INPUT_SIZE = IMG_SIZE * IMG_SIZE  # 16384

HIDDEN_SIZE = 5
OUTPUT_SIZE = 1

LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 32
SILENCE_TOP_DB = 20
VAL_SPLIT = 0.2
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4

# Audio file paths
GABRIEL_TRAIN_FILE = PROJECT_GABRIEL_TRAIN_FILE
RAIZ_TRAIN_FILE = PROJECT_RAIZ_TRAIN_FILE
GABRIEL_TEST_FILE = PROJECT_GABRIEL_TEST_FILE
RAIZ_TEST_FILE = PROJECT_RAIZ_TEST_FILE

GABRIEL_LABEL = 0
RAIZ_LABEL = 1

ensure_base_dirs()
OUTPUT_DIR = str(RESULTS_DIR / "pytorch")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "trained_pytorch_model.pt")

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------
# STEP 1: LOAD & TRIM
# ---------------------------------------------------------
def load_and_convert_m4a(filepath: str) -> np.ndarray:
    """Convert .m4a to temp wav, then load with librosa at target sample rate."""
    print(f"  Loading {filepath} ...")
    audio_seg = AudioSegment.from_file(filepath, format="m4a")
    audio_seg = audio_seg.set_frame_rate(SAMPLE_RATE).set_channels(1)
    tmp_wav = filepath.replace(".m4a", "_tmp.wav")
    audio_seg.export(tmp_wav, format="wav")
    y, sr = librosa.load(tmp_wav, sr=SAMPLE_RATE, mono=True)
    os.remove(tmp_wav)
    print(f"    Raw duration: {len(y)/SAMPLE_RATE:.1f}s at {sr}Hz")
    return y


def trim_silence(y: np.ndarray) -> np.ndarray:
    """
    Two-pass silence removal:
      Pass 1 - trim leading/trailing silence
      Pass 2 - split on internal silence gaps and concatenate voiced segments
    """
    print(f"  Trimming silence (top_db={SILENCE_TOP_DB}) ...")
    y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
    intervals = librosa.effects.split(y_trimmed, top_db=SILENCE_TOP_DB)
    voiced = [y_trimmed[s:e] for s, e in intervals]

    if not voiced:
        print("  WARNING: No voiced segments found - returning original.")
        return y

    y_voiced = np.concatenate(voiced)
    print(f"    Voiced audio after trim: {len(y_voiced)/SAMPLE_RATE:.1f}s")
    return y_voiced


# ---------------------------------------------------------
# STEP 2: SEGMENT INTO 5-SECOND CLIPS
# ---------------------------------------------------------
def segment_audio(y: np.ndarray, label: int, speaker_name: str) -> list:
    """
    Slice voiced audio into non-overlapping 5-second clips.
    Trailing chunk shorter than 5s is discarded.
    Returns list of (clip_array, label) tuples.
    """
    print(f"  Segmenting {speaker_name} into {CLIP_DURATION}s clips ...")
    n_clips = len(y) // SAMPLES_PER_CLIP
    clips = [
        (y[i * SAMPLES_PER_CLIP:(i + 1) * SAMPLES_PER_CLIP], label)
        for i in range(n_clips)
    ]
    print(f"    {len(clips)} clips extracted for {speaker_name}")
    return clips


def process_speaker(filepath: str, label: int, speaker_name: str) -> list:
    """Full per-speaker pipeline: load -> trim -> segment."""
    y = load_and_convert_m4a(filepath)
    y = trim_silence(y)
    return segment_audio(y, label, speaker_name)


# ---------------------------------------------------------
# STEP 3: CLIP -> MEL SPECTROGRAM -> FEATURE VECTOR
# ---------------------------------------------------------
def clip_to_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Compute a log-Mel spectrogram, resize to IMG_SIZE x IMG_SIZE,
    and flatten to a 1D vector of length 16384.

    Per-clip normalization is intentionally NOT applied here.
    Global z-score normalization is applied after dataset assembly.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (128, time_frames)

    if mel_db.shape[1] != IMG_SIZE:
        from PIL import Image as PILImage
        img = PILImage.fromarray(mel_db)
        img = img.resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
        mel_db = np.array(img)

    return mel_db.flatten().astype(np.float32)


def clips_to_arrays(clips: list, desc: str) -> tuple:
    """Convert a list of (clip, label) tuples to numpy arrays X, y."""
    print(f"  Converting {len(clips)} {desc} clips to spectrograms ...")
    X, y = [], []
    for i, (clip, label) in enumerate(clips):
        if i % 100 == 0:
            print(f"    [{desc}] clip {i + 1}/{len(clips)} ...")
        X.append(clip_to_spectrogram(clip))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------
# STEP 4: BUILD TRAIN & TEST DATASETS
# ---------------------------------------------------------
def build_train_dataset(gabriel_clips: list, raiz_clips: list) -> tuple:
    """
    Training set: all available clips from both speakers, shuffled.
    No cap - every clip that survived silence trimming is used.
    """
    print("\n[STEP 3 - TRAIN] Building training dataset ...")
    X, y = clips_to_arrays(gabriel_clips + raiz_clips, "train")
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    print(
        f"  Training set - Gabriel: {int(np.sum(y == 0))}, "
        f"Raiz: {int(np.sum(y == 1))}, Total: {len(X)}"
    )
    return X, y


def build_test_dataset(gabriel_clips: list, raiz_clips: list) -> tuple:
    """
    Test set: balanced by capping both speakers to the smaller clip count,
    then shuffled. Prevents evaluation being skewed by unequal class sizes.
    """
    print("\n[STEP 3 - TEST] Building test dataset ...")
    cap = min(len(gabriel_clips), len(raiz_clips))
    if len(gabriel_clips) != len(raiz_clips):
        print(f"  Balancing: capping both speakers to {cap} clips")
        gabriel_clips = gabriel_clips[:cap]
        raiz_clips = raiz_clips[:cap]

    X, y = clips_to_arrays(gabriel_clips + raiz_clips, "test")
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    print(
        f"  Test set - Gabriel: {int(np.sum(y == 0))}, "
        f"Raiz: {int(np.sum(y == 1))}, Total: {len(X)}"
    )
    return X, y


def split_train_validation(X: np.ndarray, y: np.ndarray,
                           val_split: float = VAL_SPLIT) -> tuple:
    """Stratified split of training set into train/validation."""
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")

    idx0 = np.where(y == GABRIEL_LABEL)[0]
    idx1 = np.where(y == RAIZ_LABEL)[0]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    n0_val = max(1, int(len(idx0) * val_split))
    n1_val = max(1, int(len(idx1) * val_split))
    n0_val = min(n0_val, len(idx0) - 1)
    n1_val = min(n1_val, len(idx1) - 1)

    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    fit_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    np.random.shuffle(val_idx)
    np.random.shuffle(fit_idx)

    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(
        f"  Train/Val split ({int(val_split*100)}%) - "
        f"Train: {len(X_fit)} | Val: {len(X_val)}"
    )
    print(
        f"    Train classes - Gabriel: {int(np.sum(y_fit==0))}, "
        f"Raiz: {int(np.sum(y_fit==1))}"
    )
    print(
        f"    Val classes   - Gabriel: {int(np.sum(y_val==0))}, "
        f"Raiz: {int(np.sum(y_val==1))}"
    )
    return X_fit, y_fit, X_val, y_val


# ---------------------------------------------------------
# GLOBAL NORMALIZATION
# ---------------------------------------------------------
def fit_normalizer(X_train: np.ndarray) -> tuple:
    """
    Compute per-feature mean and std from the training set only.
    Features with zero std are set to std=1 to avoid division by zero.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    print(f"  Normalizer fitted on {len(X_train)} training samples.")
    print(f"  Feature mean range : [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Feature std  range : [{std.min():.3f},  {std.max():.3f}]")
    return mean, std


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization: X_norm = (X - mean) / std."""
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------
# STEP 5: PYTORCH 2-LAYER NEURAL NETWORK
# ---------------------------------------------------------
if nn is not None:
    class SpeakerMLP(nn.Module):
        """Input -> Hidden(sigmoid) -> Output(logit)."""

        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.act1 = nn.Sigmoid()
            self.fc2 = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            return x.squeeze(1)
else:
    class SpeakerMLP:
        """Placeholder when PyTorch is not installed."""
        pass


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_model(device: torch.device) -> SpeakerMLP:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = SpeakerMLP(INPUT_SIZE, HIDDEN_SIZE).to(device)
    nn.init.xavier_uniform_(model.fc1.weight)
    nn.init.zeros_(model.fc1.bias)
    nn.init.xavier_uniform_(model.fc2.weight)
    nn.init.zeros_(model.fc2.bias)
    return model


def predict_proba(model: SpeakerMLP, X: np.ndarray, device: torch.device,
                  batch_size: int = 512) -> np.ndarray:
    """Return probabilities for class 1 (Raiz)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = torch.from_numpy(X[start:end]).float().to(device)
            logits = model(xb)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(batch_probs)
    if not probs:
        return np.array([], dtype=np.float32)
    return np.concatenate(probs, axis=0).astype(np.float32)


def train_model(model: SpeakerMLP,
                X_fit: np.ndarray, y_fit: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                device: torch.device) -> tuple:
    print("\n[STEP 5] Training PyTorch neural network ...")
    print(f"  Architecture : {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    print(f"  Epochs       : {EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Train samples: {len(X_fit)}  |  Val samples: {len(X_val)}")
    print(
        f"  Early stop   : patience={EARLY_STOP_PATIENCE}, "
        f"min_delta={EARLY_STOP_MIN_DELTA}"
    )
    print(f"  Device       : {device}\n")

    X_tensor = torch.from_numpy(X_fit).float()
    y_tensor = torch.from_numpy(y_fit).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(BATCH_SIZE, len(dataset))),
        shuffle=True,
        drop_last=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)

        epoch_loss = running_loss / len(dataset)
        train_scores = predict_proba(model, X_fit, device, batch_size=1024)
        preds = (train_scores >= 0.5).astype(np.int32)
        train_accuracy = float(np.mean(preds == y_fit.astype(np.int32)))
        val_scores = predict_proba(model, X_val, device, batch_size=1024)
        val_scores = np.clip(val_scores, 1e-12, 1.0 - 1e-12)
        val_loss = float(
            -np.mean(y_val * np.log(val_scores) + (1.0 - y_val) * np.log(1.0 - val_scores))
        )
        val_preds = (val_scores >= 0.5).astype(np.int32)
        val_accuracy = float(np.mean(val_preds == y_val.astype(np.int32)))

        history["loss"].append(float(epoch_loss))
        history["accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < (best_val_loss - EARLY_STOP_MIN_DELTA):
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{EPOCHS}"
                f"  |  Loss: {epoch_loss:.6f}"
                f"  |  Train Accuracy: {train_accuracy * 100:.2f}%"
                f"  |  Val Loss: {val_loss:.6f}"
                f"  |  Val Acc: {val_accuracy * 100:.2f}%"
            )

        if stale_epochs >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    model.load_state_dict(best_state)
    print(f"\n  Training complete. Best epoch: {best_epoch}")
    return model, history, best_epoch


def save_trained_model(path: str, model: SpeakerMLP, threshold: float,
                       norm_mean: np.ndarray, norm_std: np.ndarray):
    """Persist model state + threshold + normalizer for inference."""
    artifact = {
        "state_dict": model.state_dict(),
        "threshold": float(threshold),
        "norm_mean": norm_mean.astype(np.float32),
        "norm_std": norm_std.astype(np.float32),
        "config": {
            "sample_rate": SAMPLE_RATE,
            "clip_duration": CLIP_DURATION,
            "img_size": IMG_SIZE,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
        },
    }
    torch.save(artifact, path)
    print(f"  Saved trained model: {path}")


# ---------------------------------------------------------
# STEP 6: EVALUATION & PLOTS
# ---------------------------------------------------------
def compute_roc(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Manual ROC + AUC via trapezoidal rule."""
    thresholds = np.linspace(0, 1, 500)[::-1]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    fprs, tprs = [], []

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tprs.append(tp / pos if pos > 0 else 0.0)
        fprs.append(fp / neg if neg > 0 else 0.0)

    fprs = np.array(fprs, dtype=np.float32)
    tprs = np.array(tprs, dtype=np.float32)
    order = np.argsort(fprs)
    fprs = fprs[order]
    tprs = tprs[order]
    return fprs, tprs, float(np.trapezoid(tprs, fprs))


def select_threshold_by_youden(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Select threshold maximizing Youden's J statistic."""
    thresholds = np.linspace(0, 1, 500)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    best_thresh = 0.5
    best_j = -np.inf
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tpr = tp / pos if pos > 0 else 0.0
        fpr = fp / neg if neg > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thresh = float(thresh)
    return best_thresh


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]]."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t][p] += 1
    return cm


def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PyTorch Training History", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], color="#e74c3c", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], color="#2c3e50", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(["Train", "Validation"])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["accuracy"]], color="#2980b9", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_accuracy"]], color="#16a085", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(["Train", "Validation"])
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves_torch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(fprs: np.ndarray, tprs: np.ndarray, auc: float):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs, tprs, color="#8e44ad", linewidth=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random (AUC = 0.5)")
    ax.fill_between(fprs, tprs, alpha=0.15, color="#8e44ad")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve - Voice Speaker Classifier (PyTorch)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "roc_curve_torch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray, accuracy: float):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["Gabriel (0)", "Raiz (1)"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix (Test Accuracy: {accuracy * 100:.2f}%)", fontsize=12, fontweight="bold")

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total * 100.0) if total > 0 else 0.0
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color,
            )

    for (i, j), lbl in {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}.items():
        ax.text(j, i + 0.38, lbl, ha="center", va="center", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix_torch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_attribute_heatmap(model: SpeakerMLP):
    """
    Project first-layer weight magnitudes back onto the 128x128 spectrogram grid.
    Each pixel's importance = sum of |W1[pixel, :]| across hidden neurons.
    """
    w1 = model.fc1.weight.detach().cpu().numpy().T  # (input_size, hidden_size)
    importance = np.sum(np.abs(w1), axis=1).reshape(IMG_SIZE, IMG_SIZE)

    imp_min, imp_max = importance.min(), importance.max()
    if imp_max - imp_min > 0:
        importance = (importance - imp_min) / (imp_max - imp_min)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(importance, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Relative Weight Magnitude (normalized)")
    ax.set_title(
        "Attribute Heatmap - Input Feature Importance\n"
        "(fc1 weight magnitudes projected onto spectrogram grid)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time Frames ->", fontsize=11)
    ax.set_ylabel("Mel Frequency Bins ->", fontsize=11)

    path = os.path.join(OUTPUT_DIR, "attribute_heatmap_torch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_and_save_summary(cm: np.ndarray, auc: float, y_true: np.ndarray,
                           y_pred: np.ndarray, n_train: int, n_val: int,
                           threshold: float, best_epoch: int,
                           val_accuracy: float, val_auc: float):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    prec_r = safe_div(tp, tp + fp)
    rec_r = safe_div(tp, tp + fn)
    f1_r = safe_div(2 * prec_r * rec_r, prec_r + rec_r)
    prec_g = safe_div(tn, tn + fn)
    rec_g = safe_div(tn, tn + fp)
    f1_g = safe_div(2 * prec_g * rec_g, prec_g + rec_g)

    lines = [
        "VOICE SPEAKER CLASSIFIER - PYTORCH EVALUATION REPORT",
        "=" * 60,
        f"Architecture  : {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}",
        f"Epochs        : {EPOCHS}",
        f"Learning rate : {LEARNING_RATE}",
        f"Batch size    : {BATCH_SIZE}",
        f"Best epoch    : {best_epoch}",
        f"Sample rate   : {SAMPLE_RATE} Hz",
        f"Spectrogram   : {IMG_SIZE}x{IMG_SIZE} Mel",
        f"Train samples : {n_train}",
        f"Val samples   : {n_val}",
        f"Test samples  : {total}",
        "=" * 60,
        f"Validation Accuracy : {val_accuracy * 100:.2f}%",
        f"Validation AUC      : {val_auc:.4f}",
        f"Threshold (Val J)   : {threshold:.4f}",
        f"Accuracy      : {accuracy * 100:.2f}%",
        f"AUC           : {auc:.4f}",
        "-" * 60,
        f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        f"{'Gabriel (0)':<20} {prec_g:>10.4f} {rec_g:>10.4f} {f1_g:>10.4f}",
        f"{'Raiz (1)':<20} {prec_r:>10.4f} {rec_r:>10.4f} {f1_r:>10.4f}",
        "-" * 60,
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        "=" * 60,
    ]

    print("\n" + "\n".join(lines))
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report_torch.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved: {report_path}")


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    if torch is None:
        print("ERROR: Missing dependency 'torch'.")
        if TORCH_IMPORT_ERROR is not None:
            print(f"Import error: {TORCH_IMPORT_ERROR}")
        print("Install it in your venv with:")
        print("  python -m pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  VOICE SPEAKER CLASSIFIER - PYTORCH FULL PIPELINE")
    print("=" * 60)

    required = [GABRIEL_TRAIN_FILE, RAIZ_TRAIN_FILE, GABRIEL_TEST_FILE, RAIZ_TEST_FILE]
    for fname in required:
        if not os.path.exists(fname):
            print(f"\nERROR: File not found: {fname}")
            print("  Expected audio files under: data/audio/")
            for f in required:
                print(f"    {f}")
            sys.exit(1)

    # Step 1 & 2 - train audio
    print("\n[STEP 1 & 2 - TRAIN] Processing training audio ...")
    print("\n  - Gabriel (train) -")
    gabriel_train = process_speaker(GABRIEL_TRAIN_FILE, GABRIEL_LABEL, "Gabriel")
    print("\n  - Raiz (train) -")
    raiz_train = process_speaker(RAIZ_TRAIN_FILE, RAIZ_LABEL, "Raiz")

    print(
        f"\n  Train clips - Gabriel: {len(gabriel_train)}, "
        f"Raiz: {len(raiz_train)}, "
        f"Total: {len(gabriel_train) + len(raiz_train)}"
    )
    if not gabriel_train or not raiz_train:
        print("\nERROR: A training speaker produced 0 clips. Check audio files.")
        sys.exit(1)

    # Step 1 & 2 - test audio
    print("\n[STEP 1 & 2 - TEST] Processing test audio ...")
    print("\n  - Gabriel (test) -")
    gabriel_test = process_speaker(GABRIEL_TEST_FILE, GABRIEL_LABEL, "Gabriel")
    print("\n  - Raiz (test) -")
    raiz_test = process_speaker(RAIZ_TEST_FILE, RAIZ_LABEL, "Raiz")

    print(
        f"\n  Test clips (before balance) - Gabriel: {len(gabriel_test)}, "
        f"Raiz: {len(raiz_test)}"
    )
    if not gabriel_test or not raiz_test:
        print("\nERROR: A test speaker produced 0 clips. Check audio files.")
        sys.exit(1)

    # Step 3 & 4: Build datasets
    X_train, y_train = build_train_dataset(gabriel_train, raiz_train)
    X_test, y_test = build_test_dataset(gabriel_test, raiz_test)

    print("\n[VAL] Creating train/validation split from raw training set ...")
    X_fit, y_fit, X_val, y_val = split_train_validation(X_train, y_train, VAL_SPLIT)

    # Global normalization (fit on train-fit only to avoid validation leakage)
    print("\n[NORM] Fitting global normalizer on train-fit set only ...")
    norm_mean, norm_std = fit_normalizer(X_fit)
    X_fit = apply_normalizer(X_fit, norm_mean, norm_std)
    X_val = apply_normalizer(X_val, norm_mean, norm_std)
    X_test = apply_normalizer(X_test, norm_mean, norm_std)
    print("  Normalization applied to train-fit, validation, and test sets.")

    # Train
    device = get_device()
    model = init_model(device)
    model, history, best_epoch = train_model(model, X_fit, y_fit, X_val, y_val, device)

    # Evaluate
    print("\n[STEP 6] Evaluating on test set ...")
    val_scores = predict_proba(model, X_val, device)
    threshold = select_threshold_by_youden(y_val, val_scores)
    val_pred = (val_scores >= threshold).astype(int)
    val_accuracy = float(np.mean(val_pred == y_val.astype(int)))
    _, _, val_auc = compute_roc(y_val, val_scores)
    print(f"  Selected threshold from validation set (Youden J): {threshold:.4f}")
    print(f"  Validation Accuracy: {val_accuracy * 100:.2f}%  |  Validation AUC: {val_auc:.4f}")

    y_scores = predict_proba(model, X_test, device)
    y_pred = (y_scores >= threshold).astype(int)

    fprs, tprs, auc = compute_roc(y_test, y_scores)
    cm = compute_confusion_matrix(y_test, y_pred)
    test_accuracy = float(np.mean(y_pred == y_test.astype(int)))
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%  |  AUC: {auc:.4f}")

    # Plots + report
    print("\n  Generating output plots ...")
    plot_training_curves(history)
    plot_roc_curve(fprs, tprs, auc)
    plot_confusion_matrix(cm, test_accuracy)
    plot_attribute_heatmap(model)
    print_and_save_summary(
        cm, auc, y_test, y_pred,
        n_train=len(X_fit),
        n_val=len(X_val),
        threshold=threshold,
        best_epoch=best_epoch,
        val_accuracy=val_accuracy,
        val_auc=val_auc,
    )
    save_trained_model(MODEL_PATH, model, threshold, norm_mean, norm_std)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print("    training_curves_torch.png   - Loss & accuracy over epochs")
    print("    roc_curve_torch.png         - ROC curve with AUC score")
    print("    confusion_matrix_torch.png  - Labeled confusion matrix")
    print("    attribute_heatmap_torch.png - Input feature importance map")
    print("    evaluation_report_torch.txt - Full classification report")
    print("    trained_pytorch_model.pt    - Saved model for inference")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()

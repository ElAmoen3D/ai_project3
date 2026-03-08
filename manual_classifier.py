"""
Voice Speaker Classifier — Manual 2-Layer Neural Network
=========================================================
Classifies audio samples as:
  Gabriel = 0
  Raiz    = 1

Pipeline:
  1. Trim silence/noise from raw .m4a files
  2. Segment into 5-second clips
  3. Convert each clip to a 128x128 Mel spectrogram (flattened to 16384 features)
  4. Build labeled dataset with 80/20 train-test split
  5. Train a manual 2-layer neural network (NumPy only, sigmoid activation)
  6. Evaluate: ROC Curve, AUC, Confusion Matrix, Attribute Heatmap

Libraries used per step:
  Steps 1-3: librosa, pydub, numpy, matplotlib (audio I/O and spectrogram generation)
  Steps 4-6: numpy ONLY for the neural network (no sklearn, no torch, no tensorflow)
  Evaluation plots: matplotlib (visualization only)
"""

import os
import sys
import numpy as np
import librosa
import librosa.effects
import librosa.feature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SAMPLE_RATE       = 22050          # Hz
CLIP_DURATION     = 5              # seconds
SAMPLES_PER_CLIP  = SAMPLE_RATE * CLIP_DURATION  # 110250 samples
N_MELS            = 128
IMG_SIZE          = 128            # 128x128 spectrogram
INPUT_SIZE        = IMG_SIZE * IMG_SIZE  # 16384

HIDDEN_SIZE       = 64
OUTPUT_SIZE       = 1

LEARNING_RATE     = 0.01
EPOCHS            = 100
TRAIN_RATIO       = 0.80

SILENCE_TOP_DB    = 20             # dB threshold for silence trimming
MIN_SILENCE_MS    = 300            # minimum silence duration (ms) for pydub split

GABRIEL_FILE      = "gabriel_samples.m4a"
RAIZ_FILE         = "raiz_samples.m4a"
GABRIEL_LABEL     = 0
RAIZ_LABEL        = 1

OUTPUT_DIR        = "output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────
# STEP 1 & 2: LOAD, TRIM SILENCE, SEGMENT
# ─────────────────────────────────────────────

def load_and_convert_m4a(filepath: str) -> np.ndarray:
    """Convert .m4a to wav in memory, then load with librosa."""
    print(f"  Loading {filepath} ...")
    audio_seg = AudioSegment.from_file(filepath, format="m4a")
    audio_seg = audio_seg.set_frame_rate(SAMPLE_RATE).set_channels(1)
    # Export to temp wav
    tmp_wav = filepath.replace(".m4a", "_tmp.wav")
    audio_seg.export(tmp_wav, format="wav")
    y, sr = librosa.load(tmp_wav, sr=SAMPLE_RATE, mono=True)
    os.remove(tmp_wav)
    print(f"    Loaded {len(y)/SAMPLE_RATE:.1f}s of audio at {sr}Hz")
    return y


def trim_silence(y: np.ndarray) -> np.ndarray:
    """
    Trim leading/trailing silence, then remove internal silent gaps
    by splitting on silence and concatenating voiced segments.
    Uses librosa energy-based trimming.
    """
    print(f"  Trimming silence (top_db={SILENCE_TOP_DB}) ...")
    # First pass: trim edges
    y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)

    # Second pass: split on internal silence and rejoin
    intervals = librosa.effects.split(y_trimmed, top_db=SILENCE_TOP_DB)
    voiced_segments = [y_trimmed[start:end] for start, end in intervals]

    if len(voiced_segments) == 0:
        print("  WARNING: No voiced segments found! Returning original audio.")
        return y

    y_voiced = np.concatenate(voiced_segments)
    print(f"    After trimming: {len(y_voiced)/SAMPLE_RATE:.1f}s of voiced audio")
    return y_voiced


def segment_audio(y: np.ndarray, label: int, speaker_name: str) -> list:
    """
    Slice voiced audio into non-overlapping 5-second clips.
    Discard trailing chunk if shorter than 5 seconds.
    Returns list of (clip_array, label) tuples.
    """
    print(f"  Segmenting {speaker_name} into {CLIP_DURATION}s clips ...")
    clips = []
    total_samples = len(y)
    n_clips = total_samples // SAMPLES_PER_CLIP

    for i in range(n_clips):
        start = i * SAMPLES_PER_CLIP
        end   = start + SAMPLES_PER_CLIP
        clip  = y[start:end]
        clips.append((clip, label))

    print(f"    Extracted {len(clips)} clips for {speaker_name}")
    return clips


# ─────────────────────────────────────────────
# STEP 3: AUDIO CLIP → MEL SPECTROGRAM → FEATURE VECTOR
# ─────────────────────────────────────────────

def clip_to_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Convert a raw audio clip to a 128x128 log-Mel spectrogram,
    then flatten to a 1D feature vector of length 16384.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (128, time_frames)

    # Resize time axis to exactly IMG_SIZE columns via interpolation
    if mel_db.shape[1] != IMG_SIZE:
        from PIL import Image as PILImage
        img = PILImage.fromarray(mel_db)
        img = img.resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
        mel_db = np.array(img)

    # Normalize to [0, 1]
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    if mel_max - mel_min > 0:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = mel_db - mel_min

    return mel_norm.flatten()  # 16384-dim vector


def build_dataset(gabriel_clips: list, raiz_clips: list) -> tuple:
    """
    Convert all clips to feature vectors, combine, shuffle, split 80/20.
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    """
    print("\n[STEP 3] Converting clips to Mel spectrograms ...")
    all_X = []
    all_y = []

    for i, (clip, label) in enumerate(gabriel_clips + raiz_clips):
        if i % 100 == 0:
            print(f"  Processing clip {i+1}/{len(gabriel_clips)+len(raiz_clips)} ...")
        feat = clip_to_spectrogram(clip)
        all_X.append(feat)
        all_y.append(label)

    X = np.array(all_X, dtype=np.float32)   # (N, 16384)
    y = np.array(all_y, dtype=np.float32)   # (N,)
    print(f"  Dataset shape: X={X.shape}, y={y.shape}")

    # ── STEP 4: Shuffle and 80/20 split ──────
    print("\n[STEP 4] Splitting dataset (80% train / 20% test) ...")
    indices = np.random.permutation(len(X))
    split   = int(len(X) * TRAIN_RATIO)

    train_idx = indices[:split]
    test_idx  = indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test  = X[test_idx]
    y_test  = y[test_idx]

    print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    g_train = int(np.sum(y_train == 0))
    r_train = int(np.sum(y_train == 1))
    g_test  = int(np.sum(y_test == 0))
    r_test  = int(np.sum(y_test == 1))
    print(f"  Train — Gabriel: {g_train}, Raiz: {r_train}")
    print(f"  Test  — Gabriel: {g_test},  Raiz: {r_test}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 5: MANUAL 2-LAYER NEURAL NETWORK (NumPy only)
# ─────────────────────────────────────────────
# Architecture:
#   Input (16384) → Hidden (64, sigmoid) → Output (1, sigmoid)
#
# Forward pass:
#   Z1 = X · W1 + b1
#   A1 = sigmoid(Z1)
#   Z2 = A1 · W2 + b2
#   A2 = sigmoid(Z2)   ← final probability
#
# Loss: Binary Cross-Entropy
#   L = -[y*log(A2) + (1-y)*log(1-A2)]
#
# Backprop (chain rule):
#   dL/dA2 = -(y/A2) + (1-y)/(1-A2)
#   dA2/dZ2 = sigmoid'(Z2) = A2*(1-A2)
#   dZ2/dW2 = A1
#   ...propagate back through hidden layer
# ─────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its output a = sigmoid(z)."""
    return a * (1.0 - a)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss, clipped for numerical stability."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def initialize_weights() -> dict:
    """
    Xavier (Glorot) initialization for both layers.
    W1: (INPUT_SIZE, HIDDEN_SIZE)
    b1: (1, HIDDEN_SIZE)
    W2: (HIDDEN_SIZE, OUTPUT_SIZE)
    b2: (1, OUTPUT_SIZE)
    """
    limit1 = np.sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE))
    limit2 = np.sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE))

    params = {
        "W1": np.random.uniform(-limit1, limit1, (INPUT_SIZE, HIDDEN_SIZE)),
        "b1": np.zeros((1, HIDDEN_SIZE)),
        "W2": np.random.uniform(-limit2, limit2, (HIDDEN_SIZE, OUTPUT_SIZE)),
        "b2": np.zeros((1, OUTPUT_SIZE)),
    }
    return params


def forward_pass(X: np.ndarray, params: dict) -> tuple:
    """
    Forward pass through both layers.
    Returns activations needed for backprop.
    """
    Z1 = X @ params["W1"] + params["b1"]   # (N, 64)
    A1 = sigmoid(Z1)                        # (N, 64)
    Z2 = A1 @ params["W2"] + params["b2"]  # (N, 1)
    A2 = sigmoid(Z2)                        # (N, 1)  ← output probabilities
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def backward_pass(X: np.ndarray, y: np.ndarray, params: dict, cache: dict) -> dict:
    """
    Backpropagation — manual chain rule.
    Returns dictionary of gradients for W1, b1, W2, b2.
    """
    N   = X.shape[0]
    A1  = cache["A1"]
    A2  = cache["A2"]
    y   = y.reshape(-1, 1)

    # ── Output layer gradients ──
    # dL/dZ2 = A2 - y  (combined BCE + sigmoid derivative simplification)
    dZ2 = A2 - y                              # (N, 1)
    dW2 = (A1.T @ dZ2) / N                   # (64, 1)
    db2 = np.mean(dZ2, axis=0, keepdims=True) # (1, 1)

    # ── Hidden layer gradients ──
    dA1 = dZ2 @ params["W2"].T               # (N, 64)
    dZ1 = dA1 * sigmoid_derivative(A1)       # (N, 64)
    dW1 = (X.T @ dZ1) / N                    # (16384, 64)
    db1 = np.mean(dZ1, axis=0, keepdims=True) # (1, 64)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_weights(params: dict, grads: dict) -> dict:
    """Vanilla gradient descent weight update."""
    params["W1"] -= LEARNING_RATE * grads["dW1"]
    params["b1"] -= LEARNING_RATE * grads["db1"]
    params["W2"] -= LEARNING_RATE * grads["dW2"]
    params["b2"] -= LEARNING_RATE * grads["db2"]
    return params


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    """Return probability outputs (0–1) for each sample."""
    A2, _ = forward_pass(X, params)
    return A2.flatten()


def train(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Full training loop.
    Returns trained params and history dict with loss and accuracy per epoch.
    """
    print(f"\n[STEP 5 & 6] Training neural network ...")
    print(f"  Architecture: {INPUT_SIZE} → {HIDDEN_SIZE} → {OUTPUT_SIZE}")
    print(f"  Epochs: {EPOCHS}  |  LR: {LEARNING_RATE}  |  Samples: {len(X_train)}\n")

    params  = initialize_weights()
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, EPOCHS + 1):
        # Forward
        A2, cache = forward_pass(X_train, params)

        # Loss
        loss = binary_cross_entropy(y_train, A2.flatten())

        # Accuracy
        preds    = (A2.flatten() >= 0.5).astype(int)
        accuracy = np.mean(preds == y_train.astype(int))

        history["loss"].append(loss)
        history["accuracy"].append(accuracy)

        # Backward
        grads  = backward_pass(X_train, y_train, params, cache)
        params = update_weights(params, grads)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS}  |  Loss: {loss:.6f}  |  Accuracy: {accuracy*100:.2f}%")

    print("\n  Training complete.")
    return params, history


# ─────────────────────────────────────────────
# EVALUATION & PLOTS
# ─────────────────────────────────────────────

def compute_roc(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Manually compute ROC curve points and AUC (trapezoidal rule).
    No sklearn — pure NumPy.
    """
    thresholds = np.linspace(0, 1, 500)[::-1]
    tprs, fprs = [], []

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tpr = tp / pos if pos > 0 else 0
        fpr = fp / neg if neg > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)

    fprs = np.array(fprs)
    tprs = np.array(tprs)
    auc  = np.trapz(tprs, fprs)  # trapezoidal rule

    return fprs, tprs, abs(auc)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t][p] += 1
    return cm


def plot_training_curves(history: dict):
    """Plot loss and accuracy curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], color="#e74c3c", linewidth=2)
    axes[0].set_title("Training Loss (Binary Cross-Entropy)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["accuracy"]], color="#2980b9", linewidth=2)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(fprs: np.ndarray, tprs: np.ndarray, auc: float):
    """Plot ROC curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs, tprs, color="#8e44ad", linewidth=2.5,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1.5, label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fprs, tprs, alpha=0.15, color="#8e44ad")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Voice Speaker Classifier", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray, accuracy: float):
    """Plot labeled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["Gabriel (0)", "Raiz (1)"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix  (Test Accuracy: {accuracy*100:.2f}%)",
                 fontsize=12, fontweight="bold")

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = count / total * 100
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color=color)

    cell_labels = {(0,0): "TN", (0,1): "FP", (1,0): "FN", (1,1): "TP"}
    for (i, j), lbl in cell_labels.items():
        ax.text(j, i + 0.38, lbl, ha="center", va="center",
                fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_attribute_heatmap(params: dict):
    """
    Attribute heatmap: maps W1 weights back to spectrogram shape.

    For each input pixel, compute its total influence on the hidden layer
    by summing |W1[pixel, :]| across all hidden neurons. This gives a
    128x128 importance map showing which spectrogram regions (frequency
    bands × time) the network found most discriminative.
    """
    importance = np.sum(np.abs(params["W1"]), axis=1)  # (16384,)
    importance_map = importance.reshape(IMG_SIZE, IMG_SIZE)

    # Normalize to [0, 1]
    imp_min = importance_map.min()
    imp_max = importance_map.max()
    if imp_max - imp_min > 0:
        importance_map = (importance_map - imp_min) / (imp_max - imp_min)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(importance_map, aspect="auto", origin="lower",
                   cmap="inferno", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Relative Weight Magnitude (normalized)")

    ax.set_title("Attribute Heatmap — Input Feature Importance\n"
                 "(W1 weight magnitudes projected onto spectrogram grid)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Frames →", fontsize=11)
    ax.set_ylabel("Mel Frequency Bins →", fontsize=11)

    path = os.path.join(OUTPUT_DIR, "attribute_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_summary(cm: np.ndarray, auc: float, y_true: np.ndarray,
                  y_pred_binary: np.ndarray, y_scores: np.ndarray):
    """Print a full classification report to console."""
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total    = len(y_true)
    accuracy = (tp + tn) / total
    precision_raiz    = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_raiz       = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_raiz           = (2 * precision_raiz * recall_raiz /
                         (precision_raiz + recall_raiz)
                         if (precision_raiz + recall_raiz) > 0 else 0)
    precision_gabriel = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_gabriel    = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_gabriel        = (2 * precision_gabriel * recall_gabriel /
                         (precision_gabriel + recall_gabriel)
                         if (precision_gabriel + recall_gabriel) > 0 else 0)

    print("\n" + "="*55)
    print("           EVALUATION SUMMARY (Test Set)")
    print("="*55)
    print(f"  Total test samples : {total}")
    print(f"  Accuracy           : {accuracy*100:.2f}%")
    print(f"  AUC                : {auc:.4f}")
    print("-"*55)
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'Gabriel (0)':<20} {precision_gabriel:>10.4f} {recall_gabriel:>10.4f} {f1_gabriel:>10.4f}")
    print(f"  {'Raiz (1)':<20} {precision_raiz:>10.4f} {recall_raiz:>10.4f} {f1_raiz:>10.4f}")
    print("-"*55)
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("="*55)

    # Save to file
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("VOICE SPEAKER CLASSIFIER — EVALUATION REPORT\n")
        f.write("="*55 + "\n")
        f.write(f"Architecture : {INPUT_SIZE} → {HIDDEN_SIZE} → {OUTPUT_SIZE}\n")
        f.write(f"Epochs       : {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Sample rate  : {SAMPLE_RATE} Hz\n")
        f.write(f"Spectrogram  : {IMG_SIZE}x{IMG_SIZE} Mel\n")
        f.write("="*55 + "\n")
        f.write(f"Total test samples : {total}\n")
        f.write(f"Accuracy           : {accuracy*100:.2f}%\n")
        f.write(f"AUC                : {auc:.4f}\n")
        f.write("-"*55 + "\n")
        f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write(f"{'Gabriel (0)':<20} {precision_gabriel:>10.4f} {recall_gabriel:>10.4f} {f1_gabriel:>10.4f}\n")
        f.write(f"{'Raiz (1)':<20} {precision_raiz:>10.4f} {recall_raiz:>10.4f} {f1_raiz:>10.4f}\n")
        f.write("-"*55 + "\n")
        f.write(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}\n")
        f.write("="*55 + "\n")
    print(f"\n  Report saved: {report_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  VOICE SPEAKER CLASSIFIER — FULL PIPELINE")
    print("=" * 55)

    # ── Verify audio files exist ──────────────
    for fname in [GABRIEL_FILE, RAIZ_FILE]:
        if not os.path.exists(fname):
            print(f"\nERROR: File not found: {fname}")
            print("  Please place the .m4a files in the same directory as this script.")
            sys.exit(1)

    # ── STEP 1 & 2: Load, trim, segment ──────
    print(f"\n[STEP 1] Loading and trimming audio files ...")

    print(f"\n  — Gabriel —")
    y_gabriel = load_and_convert_m4a(GABRIEL_FILE)
    y_gabriel = trim_silence(y_gabriel)
    gabriel_clips = segment_audio(y_gabriel, GABRIEL_LABEL, "Gabriel")

    print(f"\n  — Raiz —")
    y_raiz = load_and_convert_m4a(RAIZ_FILE)
    y_raiz = trim_silence(y_raiz)
    raiz_clips = segment_audio(y_raiz, RAIZ_LABEL, "Raiz")

    print(f"\n[STEP 2] Segmentation complete.")
    print(f"  Gabriel clips: {len(gabriel_clips)}")
    print(f"  Raiz clips   : {len(raiz_clips)}")
    total_clips = len(gabriel_clips) + len(raiz_clips)
    print(f"  Total clips  : {total_clips}")

    if len(gabriel_clips) == 0 or len(raiz_clips) == 0:
        print("\nERROR: One or both speakers produced 0 clips. Check audio files.")
        sys.exit(1)

    # ── STEP 3 & 4: Spectrograms + dataset split ──
    X_train, X_test, y_train, y_test = build_dataset(gabriel_clips, raiz_clips)

    # ── STEP 5: Train ─────────────────────────
    params, history = train(X_train, y_train)

    # ── STEP 6: Evaluate ─────────────────────
    print("\n[STEP 6] Evaluating on test set ...")

    y_scores = predict(X_test, params)
    y_pred   = (y_scores >= 0.5).astype(int)

    # ROC + AUC (manual, no sklearn)
    fprs, tprs, auc = compute_roc(y_test, y_scores)

    # Confusion matrix
    cm = compute_confusion_matrix(y_test, y_pred)

    test_accuracy = np.mean(y_pred == y_test.astype(int))
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%  |  AUC: {auc:.4f}")

    # ── Generate all plots ────────────────────
    print("\n  Generating output plots ...")
    plot_training_curves(history)
    plot_roc_curve(fprs, tprs, auc)
    plot_confusion_matrix(cm, test_accuracy)
    plot_attribute_heatmap(params)

    # ── Print + save text summary ─────────────
    print_summary(cm, auc, y_test, y_pred, y_scores)

    print(f"\n  All outputs saved to: ./{OUTPUT_DIR}/")
    print("  Files:")
    print("    training_curves.png   — Loss & accuracy over epochs")
    print("    roc_curve.png         — ROC curve with AUC score")
    print("    confusion_matrix.png  — Labeled confusion matrix")
    print("    attribute_heatmap.png — Input feature importance map")
    print("    evaluation_report.txt — Full classification report")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
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
  4. Build labeled dataset:
       - TRAIN: 100% of gabriel_samples.m4a + raiz_samples.m4a (all available clips)
       - TEST:  gabriel_test.m4a + raiz_test.m4a (balanced — capped to smaller class)
  5. Train a manual 2-layer neural network (NumPy only, sigmoid activation)
  6. Evaluate: ROC Curve, AUC, Confusion Matrix, Attribute Heatmap

Libraries used per step:
  Steps 1-3: librosa, pydub, numpy, matplotlib (audio I/O and spectrogram generation)
  Steps 4-6: numpy ONLY for the neural network (no sklearn, no torch, no tensorflow)
  Evaluation plots: matplotlib (visualization only)

Train/test split rationale:
  Train and test audio come from entirely separate recordings, eliminating
  any risk of data leakage from temporally adjacent clips sharing the split.
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
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SAMPLE_RATE      = 22050
CLIP_DURATION    = 5
SAMPLES_PER_CLIP = SAMPLE_RATE * CLIP_DURATION   # 110250
N_MELS           = 128
IMG_SIZE         = 128
INPUT_SIZE       = IMG_SIZE * IMG_SIZE            # 16384

HIDDEN_SIZE      = 64
OUTPUT_SIZE      = 1

LEARNING_RATE    = 0.01
EPOCHS           = 100
SILENCE_TOP_DB   = 20

# ── Audio file paths ──────────────────────────
GABRIEL_TRAIN_FILE = "gabriel_samples.m4a"
RAIZ_TRAIN_FILE    = "raiz_samples.m4a"
GABRIEL_TEST_FILE  = "gabriel_test.m4a"
RAIZ_TEST_FILE     = "raiz_test.m4a"

GABRIEL_LABEL = 0
RAIZ_LABEL    = 1

OUTPUT_DIR = "output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────
# STEP 1: LOAD & TRIM
# ─────────────────────────────────────────────

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
      Pass 1 — trim leading/trailing silence
      Pass 2 — split on internal silence gaps and concatenate voiced segments
    """
    print(f"  Trimming silence (top_db={SILENCE_TOP_DB}) ...")
    y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
    intervals     = librosa.effects.split(y_trimmed, top_db=SILENCE_TOP_DB)
    voiced        = [y_trimmed[s:e] for s, e in intervals]

    if not voiced:
        print("  WARNING: No voiced segments found — returning original.")
        return y

    y_voiced = np.concatenate(voiced)
    print(f"    Voiced audio after trim: {len(y_voiced)/SAMPLE_RATE:.1f}s")
    return y_voiced


# ─────────────────────────────────────────────
# STEP 2: SEGMENT INTO 5-SECOND CLIPS
# ─────────────────────────────────────────────

def segment_audio(y: np.ndarray, label: int, speaker_name: str) -> list:
    """
    Slice voiced audio into non-overlapping 5-second clips.
    Trailing chunk shorter than 5s is discarded.
    Returns list of (clip_array, label) tuples.
    """
    print(f"  Segmenting {speaker_name} into {CLIP_DURATION}s clips ...")
    n_clips = len(y) // SAMPLES_PER_CLIP
    clips   = [(y[i*SAMPLES_PER_CLIP:(i+1)*SAMPLES_PER_CLIP], label)
               for i in range(n_clips)]
    print(f"    {len(clips)} clips extracted for {speaker_name}")
    return clips


def process_speaker(filepath: str, label: int, speaker_name: str) -> list:
    """Full per-speaker pipeline: load → trim → segment."""
    y = load_and_convert_m4a(filepath)
    y = trim_silence(y)
    return segment_audio(y, label, speaker_name)


# ─────────────────────────────────────────────
# STEP 3: CLIP → MEL SPECTROGRAM → FEATURE VECTOR
# ─────────────────────────────────────────────

def clip_to_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Compute a log-Mel spectrogram, resize to IMG_SIZE x IMG_SIZE,
    normalize to [0, 1], and flatten to a 1D vector of length 16384.
    """
    mel    = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE,
                                             n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)   # (128, time_frames)

    if mel_db.shape[1] != IMG_SIZE:
        from PIL import Image as PILImage
        img    = PILImage.fromarray(mel_db)
        img    = img.resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
        mel_db = np.array(img)

    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 0:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_db = mel_db - mel_min

    return mel_db.flatten().astype(np.float32)


def clips_to_arrays(clips: list, desc: str) -> tuple:
    """Convert a list of (clip, label) tuples to numpy arrays X, y."""
    print(f"  Converting {len(clips)} {desc} clips to spectrograms ...")
    X, y = [], []
    for i, (clip, label) in enumerate(clips):
        if i % 100 == 0:
            print(f"    [{desc}] clip {i+1}/{len(clips)} ...")
        X.append(clip_to_spectrogram(clip))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────
# STEP 4: BUILD TRAIN & TEST DATASETS
# ─────────────────────────────────────────────

def build_train_dataset(gabriel_clips: list, raiz_clips: list) -> tuple:
    """
    Training set: all available clips from both speakers, shuffled.
    No cap — every clip that survived silence trimming is used.
    """
    print("\n[STEP 3 — TRAIN] Building training dataset ...")
    X, y = clips_to_arrays(gabriel_clips + raiz_clips, "train")
    idx  = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    print(f"  Training set — Gabriel: {int(np.sum(y==0))}, "
          f"Raiz: {int(np.sum(y==1))}, Total: {len(X)}")
    return X, y


def build_test_dataset(gabriel_clips: list, raiz_clips: list) -> tuple:
    """
    Test set: balanced by capping both speakers to the smaller clip count,
    then shuffled. Prevents evaluation being skewed by unequal class sizes.
    """
    print("\n[STEP 3 — TEST] Building test dataset ...")
    cap = min(len(gabriel_clips), len(raiz_clips))
    if len(gabriel_clips) != len(raiz_clips):
        print(f"  Balancing: capping both speakers to {cap} clips")
        gabriel_clips = gabriel_clips[:cap]
        raiz_clips    = raiz_clips[:cap]

    X, y = clips_to_arrays(gabriel_clips + raiz_clips, "test")
    idx  = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    print(f"  Test set — Gabriel: {int(np.sum(y==0))}, "
          f"Raiz: {int(np.sum(y==1))}, Total: {len(X)}")
    return X, y


# ─────────────────────────────────────────────
# STEP 5: MANUAL 2-LAYER NEURAL NETWORK (NumPy only)
# ─────────────────────────────────────────────
# Architecture:
#   Input (16384) → Hidden (64, sigmoid) → Output (1, sigmoid)
#
# Forward pass:
#   Z1 = X · W1 + b1        shape: (N, 64)
#   A1 = sigmoid(Z1)         shape: (N, 64)
#   Z2 = A1 · W2 + b2        shape: (N, 1)
#   A2 = sigmoid(Z2)         shape: (N, 1)  ← output probability
#
# Loss: Binary Cross-Entropy
#   L = -mean[ y*log(A2) + (1-y)*log(1-A2) ]
#
# Backprop:
#   dZ2 = A2 - y                          (BCE + sigmoid combined)
#   dW2 = A1.T @ dZ2 / N
#   db2 = mean(dZ2)
#   dA1 = dZ2 @ W2.T
#   dZ1 = dA1 * A1*(1-A1)
#   dW1 = X.T @ dZ1 / N
#   db1 = mean(dZ1)
# ─────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps    = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def initialize_weights() -> dict:
    """Xavier (Glorot) uniform initialization."""
    lim1 = np.sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE))
    lim2 = np.sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
    return {
        "W1": np.random.uniform(-lim1, lim1, (INPUT_SIZE, HIDDEN_SIZE)),
        "b1": np.zeros((1, HIDDEN_SIZE)),
        "W2": np.random.uniform(-lim2, lim2, (HIDDEN_SIZE, OUTPUT_SIZE)),
        "b2": np.zeros((1, OUTPUT_SIZE)),
    }


def forward_pass(X: np.ndarray, params: dict) -> tuple:
    Z1 = X @ params["W1"] + params["b1"]
    A1 = sigmoid(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)
    return A2, {"A1": A1, "A2": A2}


def backward_pass(X: np.ndarray, y: np.ndarray,
                  params: dict, cache: dict) -> dict:
    N  = X.shape[0]
    A1 = cache["A1"]
    A2 = cache["A2"]
    y  = y.reshape(-1, 1)

    dZ2 = A2 - y
    dW2 = (A1.T @ dZ2) / N
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (X.T @ dZ1) / N
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_weights(params: dict, grads: dict) -> dict:
    params["W1"] -= LEARNING_RATE * grads["dW1"]
    params["b1"] -= LEARNING_RATE * grads["db1"]
    params["W2"] -= LEARNING_RATE * grads["dW2"]
    params["b2"] -= LEARNING_RATE * grads["db2"]
    return params


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    A2, _ = forward_pass(X, params)
    return A2.flatten()


def train(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    print(f"\n[STEP 5] Training neural network ...")
    print(f"  Architecture : {INPUT_SIZE} → {HIDDEN_SIZE} → {OUTPUT_SIZE}")
    print(f"  Epochs       : {EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"  Train samples: {len(X_train)}\n")

    params  = initialize_weights()
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, EPOCHS + 1):
        A2, cache = forward_pass(X_train, params)
        loss      = binary_cross_entropy(y_train, A2.flatten())
        preds     = (A2.flatten() >= 0.5).astype(int)
        accuracy  = np.mean(preds == y_train.astype(int))

        history["loss"].append(loss)
        history["accuracy"].append(accuracy)

        grads  = backward_pass(X_train, y_train, params, cache)
        params = update_weights(params, grads)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS}"
                  f"  |  Loss: {loss:.6f}"
                  f"  |  Train Accuracy: {accuracy*100:.2f}%")

    print("\n  Training complete.")
    return params, history


# ─────────────────────────────────────────────
# STEP 6: EVALUATION & PLOTS
# ─────────────────────────────────────────────

def compute_roc(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Manual ROC + AUC via trapezoidal rule. No sklearn."""
    thresholds = np.linspace(0, 1, 500)[::-1]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    fprs, tprs = [], []

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tprs.append(tp / pos if pos > 0 else 0)
        fprs.append(fp / neg if neg > 0 else 0)

    fprs = np.array(fprs)
    tprs = np.array(tprs)
    return fprs, tprs, abs(np.trapz(tprs, fprs))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]]."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t][p] += 1
    return cm


def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], color="#e74c3c", linewidth=2)
    axes[0].set_title("Training Loss (Binary Cross-Entropy)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["accuracy"]],
                 color="#2980b9", linewidth=2)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim([0, 105]); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(fprs: np.ndarray, tprs: np.ndarray, auc: float):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs, tprs, color="#8e44ad", linewidth=2.5,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0,1], [0,1], color="gray", linestyle="--",
            linewidth=1.5, label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fprs, tprs, alpha=0.15, color="#8e44ad")
    ax.set_xlim([0,1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Voice Speaker Classifier",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11); ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray, accuracy: float):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["Gabriel (0)", "Raiz (1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix  (Test Accuracy: {accuracy*100:.2f}%)",
                 fontsize=12, fontweight="bold")

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({count/total*100:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color=color)

    for (i,j), lbl in {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}.items():
        ax.text(j, i+0.38, lbl, ha="center", va="center",
                fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_attribute_heatmap(params: dict):
    """
    Project W1 weight magnitudes back onto the 128x128 spectrogram grid.
    Each pixel's importance = sum of |W1[pixel, :]| across all hidden neurons.
    Highlights which frequency bands x time regions drove classification.
    """
    importance = np.sum(np.abs(params["W1"]), axis=1).reshape(IMG_SIZE, IMG_SIZE)
    imp_min, imp_max = importance.min(), importance.max()
    if imp_max - imp_min > 0:
        importance = (importance - imp_min) / (imp_max - imp_min)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(importance, aspect="auto", origin="lower",
                   cmap="inferno", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Relative Weight Magnitude (normalized)")
    ax.set_title("Attribute Heatmap — Input Feature Importance\n"
                 "(W1 weight magnitudes projected onto spectrogram grid)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Frames →", fontsize=11)
    ax.set_ylabel("Mel Frequency Bins →", fontsize=11)

    path = os.path.join(OUTPUT_DIR, "attribute_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def print_and_save_summary(cm: np.ndarray, auc: float,
                           y_true: np.ndarray, y_pred: np.ndarray,
                           n_train: int):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total    = len(y_true)
    accuracy = (tp + tn) / total

    def safe_div(a, b): return a / b if b > 0 else 0

    prec_r = safe_div(tp, tp + fp);  rec_r = safe_div(tp, tp + fn)
    f1_r   = safe_div(2 * prec_r * rec_r, prec_r + rec_r)
    prec_g = safe_div(tn, tn + fn);  rec_g = safe_div(tn, tn + fp)
    f1_g   = safe_div(2 * prec_g * rec_g, prec_g + rec_g)

    lines = [
        "VOICE SPEAKER CLASSIFIER — EVALUATION REPORT",
        "=" * 55,
        f"Architecture  : {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}",
        f"Epochs        : {EPOCHS}",
        f"Learning rate : {LEARNING_RATE}",
        f"Sample rate   : {SAMPLE_RATE} Hz",
        f"Spectrogram   : {IMG_SIZE}x{IMG_SIZE} Mel",
        f"Train samples : {n_train}",
        f"Test samples  : {total}",
        "=" * 55,
        f"Accuracy      : {accuracy*100:.2f}%",
        f"AUC           : {auc:.4f}",
        "-" * 55,
        f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        f"{'Gabriel (0)':<20} {prec_g:>10.4f} {rec_g:>10.4f} {f1_g:>10.4f}",
        f"{'Raiz (1)':<20} {prec_r:>10.4f} {rec_r:>10.4f} {f1_r:>10.4f}",
        "-" * 55,
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        "=" * 55,
    ]

    print("\n" + "\n".join(lines))
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved: {report_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  VOICE SPEAKER CLASSIFIER — FULL PIPELINE")
    print("=" * 55)

    # ── Verify all four audio files exist ────
    required = [GABRIEL_TRAIN_FILE, RAIZ_TRAIN_FILE,
                GABRIEL_TEST_FILE,  RAIZ_TEST_FILE]
    for fname in required:
        if not os.path.exists(fname):
            print(f"\nERROR: File not found: {fname}")
            print("  Place all four .m4a files in the same directory as this script:")
            for f in required:
                print(f"    {f}")
            sys.exit(1)

    # ── STEP 1 & 2: Load, trim, segment — TRAIN ──
    print("\n[STEP 1 & 2 — TRAIN] Processing training audio ...")
    print("\n  — Gabriel (train) —")
    gabriel_train = process_speaker(GABRIEL_TRAIN_FILE, GABRIEL_LABEL, "Gabriel")
    print("\n  — Raiz (train) —")
    raiz_train    = process_speaker(RAIZ_TRAIN_FILE,    RAIZ_LABEL,    "Raiz")

    print(f"\n  Train clips — Gabriel: {len(gabriel_train)}, "
          f"Raiz: {len(raiz_train)}, "
          f"Total: {len(gabriel_train)+len(raiz_train)}")

    if not gabriel_train or not raiz_train:
        print("\nERROR: A training speaker produced 0 clips. Check audio files.")
        sys.exit(1)

    # ── STEP 1 & 2: Load, trim, segment — TEST ───
    print("\n[STEP 1 & 2 — TEST] Processing test audio ...")
    print("\n  — Gabriel (test) —")
    gabriel_test = process_speaker(GABRIEL_TEST_FILE, GABRIEL_LABEL, "Gabriel")
    print("\n  — Raiz (test) —")
    raiz_test    = process_speaker(RAIZ_TEST_FILE,    RAIZ_LABEL,    "Raiz")

    print(f"\n  Test clips (before balance) — Gabriel: {len(gabriel_test)}, "
          f"Raiz: {len(raiz_test)}")

    if not gabriel_test or not raiz_test:
        print("\nERROR: A test speaker produced 0 clips. Check audio files.")
        sys.exit(1)

    # ── STEP 3 & 4: Build datasets ────────────
    X_train, y_train = build_train_dataset(gabriel_train, raiz_train)
    X_test,  y_test  = build_test_dataset(gabriel_test,  raiz_test)

    # ── STEP 5: Train ─────────────────────────
    params, history = train(X_train, y_train)

    # ── STEP 6: Evaluate ─────────────────────
    print("\n[STEP 6] Evaluating on test set ...")
    y_scores = predict(X_test, params)
    y_pred   = (y_scores >= 0.5).astype(int)

    fprs, tprs, auc = compute_roc(y_test, y_scores)
    cm              = compute_confusion_matrix(y_test, y_pred)
    test_accuracy   = np.mean(y_pred == y_test.astype(int))

    print(f"  Test Accuracy: {test_accuracy*100:.2f}%  |  AUC: {auc:.4f}")

    # ── Plots ─────────────────────────────────
    print("\n  Generating output plots ...")
    plot_training_curves(history)
    plot_roc_curve(fprs, tprs, auc)
    plot_confusion_matrix(cm, test_accuracy)
    plot_attribute_heatmap(params)

    print_and_save_summary(cm, auc, y_test, y_pred, n_train=len(X_train))

    print(f"\n  All outputs saved to: ./{OUTPUT_DIR}/")
    print("    training_curves.png   — Loss & accuracy over epochs")
    print("    roc_curve.png         — ROC curve with AUC score")
    print("    confusion_matrix.png  — Labeled confusion matrix")
    print("    attribute_heatmap.png — Input feature importance map")
    print("    evaluation_report.txt — Full classification report")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
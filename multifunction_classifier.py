"""
Voice Speaker Classifier — Manual 2-Layer Neural Network
=========================================================
Classifies audio samples as:
  Gabriel = 0
  Raiz    = 1

Runs one full train/test cycle per hidden-layer activation function:
  - Sigmoid
  - ReLU
  - Tanh
  - Leaky ReLU (alpha=0.01)

Output layer is always sigmoid (binary classification).
Each run saves its results to a dedicated subfolder under output_results/.
A final comparison table is printed after all runs complete.

Pipeline per run:
  1. Trim silence/noise from raw .m4a files
  2. Segment into 5-second clips
  3. Convert each clip to a 128x128 Mel spectrogram (flattened to 16384 features)
  4. Build labeled dataset:
       - TRAIN: 100% of gabriel_samples.m4a + raiz_samples.m4a (all clips)
       - TEST:  gabriel_test.m4a + raiz_test.m4a (balanced — capped to smaller class)
       - Global z-score normalization: fit on train, applied to both
  5. Train manual 2-layer neural network (NumPy only)
  6. Evaluate: ROC Curve, AUC, Confusion Matrix, Attribute Heatmap

Libraries:
  Steps 1-3 : librosa, pydub, PIL, matplotlib (preprocessing only)
  Steps 4-6 : NumPy ONLY for the model (no sklearn, torch, tensorflow)
  Plots     : matplotlib (visualization only)
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
LEAKY_RELU_ALPHA = 0.01

# ── Audio file paths ──────────────────────────
GABRIEL_TRAIN_FILE = "gabriel_samples.m4a"
RAIZ_TRAIN_FILE    = "raiz_samples.m4a"
GABRIEL_TEST_FILE  = "gabriel_test.m4a"
RAIZ_TEST_FILE     = "raiz_test.m4a"

GABRIEL_LABEL = 0
RAIZ_LABEL    = 1

BASE_OUTPUT_DIR = "output_results"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ── Activation functions to benchmark ────────
ACTIVATION_FUNCTIONS = ["sigmoid", "relu", "tanh", "leaky_relu"]

SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────
# ACTIVATION FUNCTIONS & DERIVATIVES
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


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of ReLU given pre-activation output a = relu(z)."""
    return (a > 0).astype(np.float32)


def tanh_act(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of tanh given its output a = tanh(z)."""
    return 1.0 - a ** 2


def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, z, LEAKY_RELU_ALPHA * z)

def leaky_relu_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of Leaky ReLU given its output."""
    return np.where(a >= 0, 1.0, LEAKY_RELU_ALPHA)


# Dispatch tables — look up by name
ACTIVATION_FN = {
    "sigmoid":    sigmoid,
    "relu":       relu,
    "tanh":       tanh_act,
    "leaky_relu": leaky_relu,
}

ACTIVATION_DERIV = {
    "sigmoid":    sigmoid_derivative,
    "relu":       relu_derivative,
    "tanh":       tanh_derivative,
    "leaky_relu": leaky_relu_derivative,
}

# Display names for plots and reports
ACTIVATION_LABELS = {
    "sigmoid":    "Sigmoid",
    "relu":       "ReLU",
    "tanh":       "Tanh",
    "leaky_relu": f"Leaky ReLU (α={LEAKY_RELU_ALPHA})",
}


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
    and flatten to a 1D vector of length 16384.

    Per-clip normalization is intentionally NOT applied here.
    Global z-score normalization (fit on training set, applied to both
    train and test) is performed after the full dataset is assembled.
    """
    mel    = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE,
                                             n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)   # (128, time_frames)

    if mel_db.shape[1] != IMG_SIZE:
        from PIL import Image as PILImage
        img    = PILImage.fromarray(mel_db)
        img    = img.resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
        mel_db = np.array(img)

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
    then shuffled.
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
# GLOBAL NORMALIZATION
# ─────────────────────────────────────────────

def fit_normalizer(X_train: np.ndarray) -> tuple:
    """
    Compute per-feature mean and std from training set only.
    Features with zero std get std=1 to avoid division by zero.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std  = np.where(std == 0, 1.0, std)
    print(f"  Normalizer fitted — mean range: [{mean.min():.3f}, {mean.max():.3f}]"
          f"  std range: [{std.min():.3f}, {std.max():.3f}]")
    return mean, std


def apply_normalizer(X: np.ndarray, mean: np.ndarray,
                     std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization: (X - mean) / std."""
    return ((X - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────
# STEP 5: MANUAL 2-LAYER NEURAL NETWORK (NumPy only)
# ─────────────────────────────────────────────
# Architecture:
#   Input (16384) → Hidden (64, activation_fn) → Output (1, sigmoid)
#
# Forward pass:
#   Z1 = X · W1 + b1              shape: (N, 64)
#   A1 = hidden_activation(Z1)    shape: (N, 64)
#   Z2 = A1 · W2 + b2             shape: (N, 1)
#   A2 = sigmoid(Z2)              shape: (N, 1)  ← output probability
#
# Loss: Binary Cross-Entropy
#   L = -mean[ y*log(A2) + (1-y)*log(1-A2) ]
#
# Backprop:
#   dZ2 = A2 - y                          (BCE + sigmoid combined)
#   dW2 = A1.T @ dZ2 / N
#   db2 = mean(dZ2)
#   dA1 = dZ2 @ W2.T
#   dZ1 = dA1 * hidden_activation_deriv(A1)
#   dW1 = X.T @ dZ1 / N
#   db1 = mean(dZ1)
# ─────────────────────────────────────────────

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps    = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def initialize_weights(activation: str) -> dict:
    """
    Weight initialization strategy chosen per activation:
      - Sigmoid / Tanh : Xavier uniform  → keeps variance stable for saturating fns
      - ReLU / Leaky   : He uniform      → accounts for dead neurons by scaling up
    """
    if activation in ("relu", "leaky_relu"):
        # He uniform: limit = sqrt(6 / fan_in)
        lim1 = np.sqrt(6.0 / INPUT_SIZE)
        lim2 = np.sqrt(6.0 / HIDDEN_SIZE)
    else:
        # Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
        lim1 = np.sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE))
        lim2 = np.sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE))

    return {
        "W1": np.random.uniform(-lim1, lim1, (INPUT_SIZE, HIDDEN_SIZE)),
        "b1": np.zeros((1, HIDDEN_SIZE)),
        "W2": np.random.uniform(-lim2, lim2, (HIDDEN_SIZE, OUTPUT_SIZE)),
        "b2": np.zeros((1, OUTPUT_SIZE)),
    }


def forward_pass(X: np.ndarray, params: dict, activation: str) -> tuple:
    """
    Forward pass.
    Hidden layer uses the specified activation function.
    Output layer always uses sigmoid.
    """
    act_fn = ACTIVATION_FN[activation]

    Z1 = X @ params["W1"] + params["b1"]
    A1 = act_fn(Z1)                          # hidden activation
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)                         # output always sigmoid
    return A2, {"Z1": Z1, "A1": A1, "A2": A2}


def backward_pass(X: np.ndarray, y: np.ndarray, params: dict,
                  cache: dict, activation: str) -> dict:
    """
    Backpropagation — manual chain rule.
    Hidden layer derivative uses the activation-specific function.
    """
    deriv_fn = ACTIVATION_DERIV[activation]
    N  = X.shape[0]
    A1 = cache["A1"]
    A2 = cache["A2"]
    y  = y.reshape(-1, 1)

    # ── Output layer ──
    dZ2 = A2 - y                              # (N, 1)
    dW2 = (A1.T @ dZ2) / N                   # (64, 1)
    db2 = np.mean(dZ2, axis=0, keepdims=True) # (1, 1)

    # ── Hidden layer ──
    dA1 = dZ2 @ params["W2"].T               # (N, 64)

    # ReLU/Leaky ReLU derivatives need Z1 (pre-activation), not A1.
    # Sigmoid/Tanh derivatives are expressed in terms of A1 (post-activation).
    if activation in ("relu", "leaky_relu"):
        dZ1 = dA1 * deriv_fn(cache["Z1"])
    else:
        dZ1 = dA1 * deriv_fn(A1)

    dW1 = (X.T @ dZ1) / N                    # (16384, 64)
    db1 = np.mean(dZ1, axis=0, keepdims=True) # (1, 64)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_weights(params: dict, grads: dict) -> dict:
    params["W1"] -= LEARNING_RATE * grads["dW1"]
    params["b1"] -= LEARNING_RATE * grads["db1"]
    params["W2"] -= LEARNING_RATE * grads["dW2"]
    params["b2"] -= LEARNING_RATE * grads["db2"]
    return params


def predict(X: np.ndarray, params: dict, activation: str) -> np.ndarray:
    A2, _ = forward_pass(X, params, activation)
    return A2.flatten()


def train(X_train: np.ndarray, y_train: np.ndarray,
          activation: str) -> tuple:
    label = ACTIVATION_LABELS[activation]
    print(f"\n[STEP 5] Training — hidden activation: {label}")
    print(f"  Architecture : {INPUT_SIZE} → {HIDDEN_SIZE}({label}) → {OUTPUT_SIZE}(Sigmoid)")
    print(f"  Epochs       : {EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"  Train samples: {len(X_train)}\n")

    np.random.seed(SEED)   # reset seed per run for fair comparison
    params  = initialize_weights(activation)
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, EPOCHS + 1):
        A2, cache = forward_pass(X_train, params, activation)
        loss      = binary_cross_entropy(y_train, A2.flatten())
        preds     = (A2.flatten() >= 0.5).astype(int)
        accuracy  = np.mean(preds == y_train.astype(int))

        history["loss"].append(loss)
        history["accuracy"].append(accuracy)

        grads  = backward_pass(X_train, y_train, params, cache, activation)
        params = update_weights(params, grads)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS}"
                  f"  |  Loss: {loss:.6f}"
                  f"  |  Train Accuracy: {accuracy*100:.2f}%")

    print(f"\n  Training complete ({label}).")
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
    return fprs, tprs, abs(np.trapezoid(tprs, fprs))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]]."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t][p] += 1
    return cm


def plot_training_curves(history: dict, activation: str, out_dir: str):
    label  = ACTIVATION_LABELS[activation]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History — {label}", fontsize=14, fontweight="bold")
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
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(fprs: np.ndarray, tprs: np.ndarray, auc: float,
                   activation: str, out_dir: str):
    label = ACTIVATION_LABELS[activation]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs, tprs, color="#8e44ad", linewidth=2.5,
            label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0,1], [0,1], color="gray", linestyle="--",
            linewidth=1.5, label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fprs, tprs, alpha=0.15, color="#8e44ad")
    ax.set_xlim([0,1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(f"ROC Curve — {label}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11); ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray, accuracy: float,
                          activation: str, out_dir: str):
    label = ACTIVATION_LABELS[activation]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0,1]); ax.set_xticklabels(["Gabriel (0)", "Raiz (1)"], fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Gabriel (0)", "Raiz (1)"], fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {label}\n(Test Accuracy: {accuracy*100:.2f}%)",
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
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_attribute_heatmap(params: dict, activation: str, out_dir: str):
    """
    Project W1 weight magnitudes back onto the 128x128 spectrogram grid.
    Each pixel's importance = sum of |W1[pixel, :]| across all hidden neurons.
    """
    label = ACTIVATION_LABELS[activation]
    importance = np.sum(np.abs(params["W1"]), axis=1).reshape(IMG_SIZE, IMG_SIZE)
    imp_min, imp_max = importance.min(), importance.max()
    if imp_max - imp_min > 0:
        importance = (importance - imp_min) / (imp_max - imp_min)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(importance, aspect="auto", origin="lower",
                   cmap="inferno", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Relative Weight Magnitude (normalized)")
    ax.set_title(f"Attribute Heatmap — {label}\n"
                 "(W1 weight magnitudes projected onto spectrogram grid)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Frames →", fontsize=11)
    ax.set_ylabel("Mel Frequency Bins →", fontsize=11)

    path = os.path.join(out_dir, "attribute_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def save_run_report(cm: np.ndarray, auc: float, y_true: np.ndarray,
                    y_pred: np.ndarray, n_train: int,
                    activation: str, out_dir: str) -> dict:
    """Save per-run evaluation report and return metrics dict."""
    label = ACTIVATION_LABELS[activation]
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total    = len(y_true)
    accuracy = (tp + tn) / total

    def safe_div(a, b): return a / b if b > 0 else 0

    prec_r = safe_div(tp, tp + fp);  rec_r = safe_div(tp, tp + fn)
    f1_r   = safe_div(2 * prec_r * rec_r, prec_r + rec_r)
    prec_g = safe_div(tn, tn + fn);  rec_g = safe_div(tn, tn + fp)
    f1_g   = safe_div(2 * prec_g * rec_g, prec_g + rec_g)

    lines = [
        f"VOICE SPEAKER CLASSIFIER — EVALUATION REPORT",
        f"Hidden activation : {label}",
        "=" * 55,
        f"Architecture  : {INPUT_SIZE} -> {HIDDEN_SIZE}({label}) -> {OUTPUT_SIZE}(Sigmoid)",
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
    report_path = os.path.join(out_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report saved: {report_path}")

    return {
        "activation": label,
        "accuracy":   accuracy,
        "auc":        auc,
        "f1_gabriel": f1_g,
        "f1_raiz":    f1_r,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_comparison_table(results: list):
    """Print a final summary table comparing all activation function runs."""
    sep = "=" * 75
    print(f"\n\n{sep}")
    print("  FINAL COMPARISON — ALL ACTIVATION FUNCTIONS")
    print(sep)
    print(f"  {'Activation':<22} {'Accuracy':>10} {'AUC':>8} "
          f"{'F1 Gabriel':>12} {'F1 Raiz':>10}")
    print("-" * 75)

    # Sort by accuracy descending
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  {r['activation']:<22} "
              f"{r['accuracy']*100:>9.2f}% "
              f"{r['auc']:>8.4f} "
              f"{r['f1_gabriel']:>12.4f} "
              f"{r['f1_raiz']:>10.4f}")

    print(sep)
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\n  Best accuracy : {best['activation']} ({best['accuracy']*100:.2f}%)")
    best_auc = max(results, key=lambda x: x["auc"])
    print(f"  Best AUC      : {best_auc['activation']} ({best_auc['auc']:.4f})")
    print(sep + "\n")

    # Save comparison to file
    comp_path = os.path.join(BASE_OUTPUT_DIR, "comparison_summary.txt")
    with open(comp_path, "w") as f:
        f.write("FINAL COMPARISON — ALL ACTIVATION FUNCTIONS\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'Activation':<22} {'Accuracy':>10} {'AUC':>8} "
                f"{'F1 Gabriel':>12} {'F1 Raiz':>10}\n")
        f.write("-" * 75 + "\n")
        for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
            f.write(f"{r['activation']:<22} "
                    f"{r['accuracy']*100:>9.2f}% "
                    f"{r['auc']:>8.4f} "
                    f"{r['f1_gabriel']:>12.4f} "
                    f"{r['f1_raiz']:>10.4f}\n")
        f.write("=" * 75 + "\n")
        f.write(f"Best accuracy : {best['activation']} ({best['accuracy']*100:.2f}%)\n")
        f.write(f"Best AUC      : {best_auc['activation']} ({best_auc['auc']:.4f})\n")
    print(f"  Comparison saved: {comp_path}")


# ─────────────────────────────────────────────
# SINGLE ACTIVATION RUN
# ─────────────────────────────────────────────

def run_one(activation: str, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray, n_train: int) -> dict:
    """
    Execute a complete train/evaluate cycle for one activation function.
    Saves all outputs to output_results/<activation_name>/.
    Returns a metrics dict for the final comparison table.
    """
    label   = ACTIVATION_LABELS[activation]
    out_dir = os.path.join(BASE_OUTPUT_DIR, activation)
    os.makedirs(out_dir, exist_ok=True)

    banner = f"  RUN: {label}  →  output_results/{activation}/"
    print(f"\n{'='*55}")
    print(banner)
    print(f"{'='*55}")

    # Train
    params, history = train(X_train, y_train, activation)

    # Evaluate
    print(f"\n[STEP 6 — {label}] Evaluating on test set ...")
    y_scores = predict(X_test, params, activation)
    y_pred   = (y_scores >= 0.5).astype(int)

    fprs, tprs, auc = compute_roc(y_test, y_scores)
    cm              = compute_confusion_matrix(y_test, y_pred)
    test_accuracy   = np.mean(y_pred == y_test.astype(int))

    print(f"  Test Accuracy: {test_accuracy*100:.2f}%  |  AUC: {auc:.4f}")

    # Plots
    print(f"\n  Saving plots to {out_dir}/ ...")
    plot_training_curves(history, activation, out_dir)
    plot_roc_curve(fprs, tprs, auc, activation, out_dir)
    plot_confusion_matrix(cm, test_accuracy, activation, out_dir)
    plot_attribute_heatmap(params, activation, out_dir)

    # Report
    return save_run_report(cm, auc, y_test, y_pred, n_train, activation, out_dir)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  VOICE SPEAKER CLASSIFIER — MULTI-ACTIVATION RUN")
    print(f"  Activations : {', '.join(ACTIVATION_LABELS[a] for a in ACTIVATION_FUNCTIONS)}")
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
    # Done once — datasets are reused across all activation runs
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
    # Built once and shared across all runs — only training differs
    X_train, y_train = build_train_dataset(gabriel_train, raiz_train)
    X_test,  y_test  = build_test_dataset(gabriel_test,  raiz_test)

    # ── Global normalization ──────────────────
    print("\n[NORM] Fitting global normalizer on training set ...")
    norm_mean, norm_std = fit_normalizer(X_train)
    X_train = apply_normalizer(X_train, norm_mean, norm_std)
    X_test  = apply_normalizer(X_test,  norm_mean, norm_std)
    print("  Normalization applied to train and test sets.")

    # ── STEPS 5 & 6: One run per activation ──
    all_results = []
    for activation in ACTIVATION_FUNCTIONS:
        result = run_one(
            activation=activation,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_train=len(X_train),
        )
        all_results.append(result)

    # ── Final comparison table ────────────────
    print_comparison_table(all_results)

    print(f"  Output structure:")
    for activation in ACTIVATION_FUNCTIONS:
        print(f"    output_results/{activation}/")
        print(f"      training_curves.png, roc_curve.png, "
              f"confusion_matrix.png, attribute_heatmap.png, evaluation_report.txt")
    print(f"    output_results/comparison_summary.txt")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
"""
Live microphone speaker test using the manual classifier pipeline.

Usage examples:
  python src/live_speaker_test.py
  python src/live_speaker_test.py --retrain
  python src/live_speaker_test.py --once
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: Missing dependency 'sounddevice'.")
    print("Install it with: pip install sounddevice")
    sys.exit(1)

import manuel_classifier as uc
from project_paths import (
    GABRIEL_TRAIN_FILE,
    GABRIEL_TEST_FILE,
    RAIZ_TRAIN_FILE,
    RAIZ_TEST_FILE,
    RESULTS_DIR,
    ensure_base_dirs,
)


ensure_base_dirs()
LIVE_RESULTS_DIR = os.path.join(str(RESULTS_DIR), "live")
os.makedirs(LIVE_RESULTS_DIR, exist_ok=True)

MODEL_CANDIDATES = [
    os.path.join(LIVE_RESULTS_DIR, "trained_live_model.npz"),
    os.path.join(LIVE_RESULTS_DIR, "trained_updated_model.npz"),
    os.path.join(str(RESULTS_DIR), "manual", "trained_updated_model.npz"),
    os.path.join(str(RESULTS_DIR), "multifunction", "trained_updated_model.npz"),
    os.path.join(str(RESULTS_DIR), "manual", "trained_manual_model.npz"),
]


def load_saved_model(path: str):
    """Load model params, threshold, normalizer stats, and optional activation."""
    if not os.path.exists(path):
        return None, None, None, None, None

    data = np.load(path)
    required = {"W1", "b1", "W2", "b2", "threshold"}
    if not required.issubset(set(data.files)):
        return None, None, None, None, None

    params = {
        "W1": data["W1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "W2": data["W2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
    }
    threshold = float(data["threshold"])

    input_size = int(params["W1"].shape[0])
    norm_mean = (
        data["norm_mean"].astype(np.float32)
        if "norm_mean" in data.files
        else np.zeros(input_size, dtype=np.float32)
    )
    norm_std = (
        data["norm_std"].astype(np.float32)
        if "norm_std" in data.files
        else np.ones(input_size, dtype=np.float32)
    )
    hidden_activation = None
    if "hidden_activation" in data.files:
        hidden_activation = str(data["hidden_activation"])
        if hidden_activation.startswith("b'") and hidden_activation.endswith("'"):
            hidden_activation = hidden_activation[2:-1]
        hidden_activation = hidden_activation.strip().lower()
    return params, threshold, norm_mean, norm_std, hidden_activation


def ensure_training_files_exist():
    required = [GABRIEL_TRAIN_FILE, RAIZ_TRAIN_FILE, GABRIEL_TEST_FILE, RAIZ_TEST_FILE]
    missing = [fname for fname in required if not os.path.exists(fname)]
    if missing:
        raise FileNotFoundError(
            "Missing required audio file(s): " + ", ".join(missing)
        )


def find_cached_model():
    for path in MODEL_CANDIDATES:
        params, threshold, norm_mean, norm_std, hidden_activation = load_saved_model(path)
        if params is not None:
            return path, params, threshold, norm_mean, norm_std, hidden_activation
    return None, None, None, None, None, None


def train_and_cache_model():
    ensure_training_files_exist()
    print("Running classifier full pipeline to train/test model...")
    uc.main()

    loaded_path, params, threshold, norm_mean, norm_std, hidden_activation = find_cached_model()
    if params is None:
        raise RuntimeError("No valid saved model found after retraining.")

    print(f"Loaded freshly trained model: {loaded_path}")
    print(f"Using threshold: {threshold:.4f}")
    if hidden_activation:
        print(f"Model hidden activation: {hidden_activation}")
    return params, threshold, norm_mean, norm_std, hidden_activation


def get_model(force_retrain: bool):
    if not force_retrain:
        loaded_path, params, threshold, norm_mean, norm_std, hidden_activation = find_cached_model()
        if params is not None:
            print(f"Loaded model cache: {loaded_path}")
            print(f"Using threshold: {threshold:.4f}")
            if hidden_activation:
                print(f"Model hidden activation: {hidden_activation}")
            return params, threshold, norm_mean, norm_std, hidden_activation

    return train_and_cache_model()


def record_clip(seconds: int, sample_rate: int) -> np.ndarray:
    n_samples = int(seconds * sample_rate)
    print(f"\nRecording for {seconds} seconds... speak now.")
    audio = sd.rec(n_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.reshape(-1)


def prepare_live_clip(raw_audio: np.ndarray, target_samples: int,
                      apply_trim_silence: bool = True) -> np.ndarray:
    trimmed = uc.trim_silence(raw_audio) if apply_trim_silence else raw_audio
    if len(trimmed) == 0:
        trimmed = raw_audio
    if len(trimmed) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if len(trimmed) >= target_samples:
        return trimmed[:target_samples].astype(np.float32)

    reps = int(np.ceil(target_samples / len(trimmed)))
    return np.tile(trimmed, reps)[:target_samples].astype(np.float32)


def classify_audio(
    raw_audio: np.ndarray,
    params: dict,
    threshold: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    target_samples: int,
    hidden_activation: str,
    apply_trim_silence: bool = True,
):
    prepared = prepare_live_clip(
        raw_audio,
        target_samples,
        apply_trim_silence=apply_trim_silence,
    )
    feat = uc.clip_to_spectrogram(prepared).reshape(1, -1)
    feat = uc.apply_normalizer(feat, norm_mean, norm_std)
    score = float(uc.predict(feat, params, hidden_activation)[0])

    pred = 1 if score >= threshold else 0
    speaker = "Raiz" if pred == 1 else "Gabriel"
    confidence = score if pred == 1 else (1.0 - score)
    margin = abs(score - threshold)
    return speaker, score, confidence, margin


def classify_clip(
    raw_audio: np.ndarray,
    params: dict,
    threshold: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    hidden_activation: str,
):
    speaker, score, confidence, margin = classify_audio(
        raw_audio,
        params,
        threshold,
        norm_mean,
        norm_std,
        int(uc.CLIP_DURATION * uc.SAMPLE_RATE),
        hidden_activation=hidden_activation,
        apply_trim_silence=True,
    )
    print(f"Prediction: {speaker}")
    print(
        f"Score(Raiz=1): {score:.4f} | Threshold: {threshold:.4f} | "
        f"Confidence: {confidence:.4f} | Margin: {margin:.4f}"
    )


def run_live_stream(
    params: dict,
    threshold: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    hidden_activation: str,
    window_seconds: int,
    hop_seconds: float,
    min_rms: float,
    min_margin: float,
    smooth_votes: int,
    print_all: bool,
    live_trim_silence: bool,
    input_latency: str,
    infer_every: int,
):
    sample_rate = uc.SAMPLE_RATE
    window_samples = int(window_seconds * sample_rate)

    if window_samples <= 0:
        raise ValueError("window_seconds must be > 0")
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be > 0")
    if infer_every <= 0:
        raise ValueError("infer_every must be >= 1")

    max_buffer_samples = window_samples * 4
    ring_buffer = np.zeros(max_buffer_samples, dtype=np.float32)
    write_pos = 0
    buffered_samples = 0
    buffer_lock = threading.Lock()

    last_label = None
    overflow_warned = False
    overflow_events = 0
    vote_buffer = deque(maxlen=max(1, smooth_votes))

    print("\nContinuous live classifier ready.")
    print(
        f"Window: {window_seconds}s | Update every: {hop_seconds:.2f}s | "
        f"Min RMS: {min_rms:.4f} | Min margin: {min_margin:.4f} | "
        f"Smoothing: {max(1, smooth_votes)} | "
        f"Live trim silence: {live_trim_silence} | "
        f"Input latency: {input_latency} | "
        f"Infer every: {infer_every} tick(s) | "
        f"Hidden activation: {hidden_activation}"
    )
    print("Press Ctrl+C to stop.\n")

    _ = classify_audio(
        np.zeros(window_samples, dtype=np.float32),
        params,
        threshold,
        norm_mean,
        norm_std,
        window_samples,
        hidden_activation=hidden_activation,
        apply_trim_silence=False,
    )

    def audio_callback(indata, frames, time_info, status):
        nonlocal write_pos, buffered_samples, overflow_events
        if status.input_overflow:
            overflow_events += 1

        chunk = indata[:, 0].copy()
        n = len(chunk)
        if n == 0:
            return

        with buffer_lock:
            if n >= max_buffer_samples:
                ring_buffer[:] = chunk[-max_buffer_samples:]
                write_pos = 0
                buffered_samples = max_buffer_samples
                return

            end_pos = write_pos + n
            if end_pos <= max_buffer_samples:
                ring_buffer[write_pos:end_pos] = chunk
            else:
                first = max_buffer_samples - write_pos
                ring_buffer[write_pos:] = chunk[:first]
                ring_buffer[:n - first] = chunk[first:]

            write_pos = (write_pos + n) % max_buffer_samples
            buffered_samples = min(max_buffer_samples, buffered_samples + n)

    def get_latest_window():
        with buffer_lock:
            if buffered_samples < window_samples:
                return None

            end = write_pos
            start = (end - window_samples) % max_buffer_samples
            if start < end:
                return ring_buffer[start:end].copy()

            return np.concatenate((ring_buffer[start:], ring_buffer[:end]), axis=0)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=0,
        latency=input_latency,
        callback=audio_callback,
    ):
        next_update = time.monotonic() + hop_seconds
        tick_index = 0

        while True:
            now = time.monotonic()
            if now < next_update:
                time.sleep(min(0.02, next_update - now))
                continue
            while next_update <= now:
                next_update += hop_seconds

            if overflow_events > 0 and not overflow_warned:
                print(
                    "WARNING: Audio input overflow detected. Predictions may be delayed. "
                    "Try higher --hop-seconds and --input-latency high."
                )
                overflow_warned = True

            latest_window = get_latest_window()
            if latest_window is None:
                continue

            tick_index += 1
            if (tick_index % infer_every) != 0:
                continue

            rms = float(np.sqrt(np.mean(latest_window ** 2)))
            if rms < min_rms:
                if print_all:
                    print("Prediction: (silence/noise) | waiting for clearer speech")
                continue

            speaker, score, confidence, margin = classify_audio(
                latest_window,
                params,
                threshold,
                norm_mean,
                norm_std,
                window_samples,
                hidden_activation=hidden_activation,
                apply_trim_silence=live_trim_silence,
            )

            if margin < min_margin:
                label = "Uncertain"
                vote_buffer.clear()
            else:
                vote_buffer.append(speaker)
                raiz_votes = sum(1 for s in vote_buffer if s == "Raiz")
                gabriel_votes = len(vote_buffer) - raiz_votes
                if raiz_votes > gabriel_votes:
                    label = "Raiz"
                elif gabriel_votes > raiz_votes:
                    label = "Gabriel"
                else:
                    label = speaker

            if print_all or label != last_label:
                print(
                    f"Prediction: {label} | "
                    f"Score(Raiz=1): {score:.4f} | "
                    f"Threshold: {threshold:.4f} | "
                    f"Confidence: {confidence:.4f} | "
                    f"Margin: {margin:.4f} | "
                    f"RMS: {rms:.4f}"
                )
                last_label = label


def main():
    parser = argparse.ArgumentParser(description="Live mic speaker test (Gabriel vs Raiz).")
    parser.add_argument("--retrain", action="store_true", help="Force retraining before mic testing.")
    parser.add_argument("--once", action="store_true", help="Record and classify one clip, then exit.")
    parser.add_argument("--seconds", type=int, default=uc.CLIP_DURATION, help=f"One-shot seconds (default: {uc.CLIP_DURATION}).")
    parser.add_argument("--window-seconds", type=int, default=uc.CLIP_DURATION, help=f"Rolling window seconds (default: {uc.CLIP_DURATION}).")
    parser.add_argument("--hop-seconds", type=float, default=1.0, help="Continuous update interval in seconds (default: 1.0).")
    parser.add_argument("--min-rms", type=float, default=0.005, help="Ignore low-energy audio below this RMS.")
    parser.add_argument("--min-margin", type=float, default=0.03, help="Label Uncertain if |score-threshold| below this.")
    parser.add_argument("--smooth-votes", type=int, default=5, help="Majority vote window size.")
    parser.add_argument("--print-all", action="store_true", help="Print every update.")
    parser.add_argument("--live-trim-silence", action="store_true", help="Apply silence trimming on each live window (slower).")
    parser.add_argument("--input-latency", choices=["low", "high"], default="high", help="Audio input latency hint.")
    parser.add_argument("--infer-every", type=int, default=1, help="Run inference every N update ticks.")
    parser.add_argument(
        "--hidden-activation",
        choices=uc.ACTIVATION_FUNCTIONS,
        default=None,
        help="Override hidden activation used for predict(). Default uses saved model metadata.",
    )
    args = parser.parse_args()

    try:
        params, threshold, norm_mean, norm_std, model_activation = get_model(force_retrain=args.retrain)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    hidden_activation = args.hidden_activation or model_activation or "tanh"
    if model_activation and args.hidden_activation and args.hidden_activation != model_activation:
        print(
            f"WARNING: Overriding saved activation '{model_activation}' "
            f"with CLI activation '{args.hidden_activation}'."
        )

    try:
        if args.once:
            raw_audio = record_clip(args.seconds, uc.SAMPLE_RATE)
            classify_clip(
                raw_audio,
                params,
                threshold,
                norm_mean,
                norm_std,
                hidden_activation=hidden_activation,
            )
            return

        run_live_stream(
            params=params,
            threshold=threshold,
            norm_mean=norm_mean,
            norm_std=norm_std,
            hidden_activation=hidden_activation,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            min_rms=args.min_rms,
            min_margin=args.min_margin,
            smooth_votes=args.smooth_votes,
            print_all=args.print_all,
            live_trim_silence=args.live_trim_silence,
            input_latency=args.input_latency,
            infer_every=args.infer_every,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

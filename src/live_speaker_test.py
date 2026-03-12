"""
Live microphone speaker test for the updated voice classifier.

Usage examples:
  python live_speaker_test.py
  python live_speaker_test.py --retrain
  python live_speaker_test.py --once
  python live_speaker_test.py --window-seconds 5 --hop-seconds 1
"""

import argparse
import importlib
import inspect
import os
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    from .project_paths import (
        GABRIEL_TRAIN_FILE as PROJECT_GABRIEL_TRAIN_FILE,
        RAIZ_TRAIN_FILE as PROJECT_RAIZ_TRAIN_FILE,
        GABRIEL_TEST_FILE as PROJECT_GABRIEL_TEST_FILE,
        RAIZ_TEST_FILE as PROJECT_RAIZ_TEST_FILE,
        RESULTS_DIR,
        ensure_base_dirs,
    )
except ImportError:
    from project_paths import (  # type: ignore
        GABRIEL_TRAIN_FILE as PROJECT_GABRIEL_TRAIN_FILE,
        RAIZ_TRAIN_FILE as PROJECT_RAIZ_TRAIN_FILE,
        GABRIEL_TEST_FILE as PROJECT_GABRIEL_TEST_FILE,
        RAIZ_TEST_FILE as PROJECT_RAIZ_TEST_FILE,
        RESULTS_DIR,
        ensure_base_dirs,
    )

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: Missing dependency 'sounddevice'.")
    print("Install it with: pip install sounddevice")
    sys.exit(1)

def load_classifier_module():
    """
    Load classifier module with fallback names so file renames don't break live test.
    Priority:
      1) $SPEAKER_CLASSIFIER_MODULE (if set)
      2) updated_classifier
      3) manuel_classifier
      4) manual_classifier
    """
    candidates = []
    env_name = os.environ.get("SPEAKER_CLASSIFIER_MODULE", "").strip()
    if env_name:
        candidates.append(env_name)
    base_candidates = [
        "updated_classifier",
        "manuel_classifier",
        "manual_classifier",
        "multifunction_classifier",
    ]
    candidates.extend(base_candidates)
    candidates.extend([f"src.{name}" for name in base_candidates])

    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            module = importlib.import_module(name)
            if name != "updated_classifier":
                print(f"Using classifier module: {name}")
            return module
        except ModuleNotFoundError:
            continue

    print("ERROR: Could not import classifier module.")
    print(
        "Expected one of: updated_classifier.py, manuel_classifier.py, "
        "manual_classifier.py, multifunction_classifier.py."
    )
    print("Or set env var SPEAKER_CLASSIFIER_MODULE=<your_module_name>.")
    sys.exit(1)


uc = load_classifier_module()

ensure_base_dirs()
DEFAULT_LIVE_MODEL_PATH = str(RESULTS_DIR / "live" / "trained_live_model.npz")
LEGACY_LIVE_MODEL_PATH = str(RESULTS_DIR / "live" / "trained_updated_model.npz")
os.makedirs(os.path.dirname(DEFAULT_LIVE_MODEL_PATH), exist_ok=True)


MODEL_PATH = getattr(uc, "MODEL_PATH", None)
if not MODEL_PATH:
    module_output_dir = getattr(uc, "OUTPUT_DIR", None) or getattr(uc, "BASE_OUTPUT_DIR", None)
    if module_output_dir:
        MODEL_PATH = os.path.join(module_output_dir, "trained_updated_model.npz")
    else:
        MODEL_PATH = DEFAULT_LIVE_MODEL_PATH


def _detect_predict_needs_activation() -> bool:
    predict_fn = getattr(uc, "predict", None)
    if predict_fn is None:
        return False
    try:
        sig = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        return False

    params = list(sig.parameters.values())
    if len(params) >= 3:
        return True
    return any(p.name == "activation" for p in params)


PREDICT_NEEDS_ACTIVATION = _detect_predict_needs_activation()
MODULE_ACTIVATIONS = list(getattr(uc, "ACTIVATION_FUNCTIONS", []))
DEFAULT_HIDDEN_ACTIVATION = (
    os.environ.get("SPEAKER_HIDDEN_ACTIVATION", "").strip()
    or ("sigmoid" if "sigmoid" in MODULE_ACTIVATIONS else (MODULE_ACTIVATIONS[0] if MODULE_ACTIVATIONS else "sigmoid"))
)


def load_saved_model(path: str):
    """Load model params, threshold, and normalizer stats from disk."""
    if hasattr(uc, "load_saved_model"):
        loaded = uc.load_saved_model(path)
        if loaded is None:
            return None, None, None, None
        if isinstance(loaded, tuple) and len(loaded) == 4:
            return loaded
        if isinstance(loaded, tuple) and len(loaded) == 2:
            params, threshold = loaded
            input_size = int(params["W1"].shape[0])
            return (
                params,
                float(threshold),
                np.zeros(input_size, dtype=np.float32),
                np.ones(input_size, dtype=np.float32),
            )

    if not os.path.exists(path):
        return None, None, None, None

    data = np.load(path)
    required = {"W1", "b1", "W2", "b2", "threshold"}
    if not required.issubset(set(data.files)):
        return None, None, None, None

    params = {
        "W1": data["W1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "W2": data["W2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
    }
    threshold = float(data["threshold"])

    input_size = int(params["W1"].shape[0])
    norm_mean = data["norm_mean"].astype(np.float32) if "norm_mean" in data.files else np.zeros(input_size, dtype=np.float32)
    norm_std = data["norm_std"].astype(np.float32) if "norm_std" in data.files else np.ones(input_size, dtype=np.float32)
    return params, threshold, norm_mean, norm_std


def ensure_training_files_exist():
    """Ensure the four source files required by updated_classifier are present."""
    required = [
        getattr(uc, "GABRIEL_TRAIN_FILE", PROJECT_GABRIEL_TRAIN_FILE),
        getattr(uc, "RAIZ_TRAIN_FILE", PROJECT_RAIZ_TRAIN_FILE),
        getattr(uc, "GABRIEL_TEST_FILE", PROJECT_GABRIEL_TEST_FILE),
        getattr(uc, "RAIZ_TEST_FILE", PROJECT_RAIZ_TEST_FILE),
    ]
    missing = [fname for fname in required if not os.path.exists(fname)]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required audio file(s) for retraining/testing: {missing_list}"
        )


def train_and_cache_model(path: str):
    """
    Run the full updated_classifier pipeline (train + test + save),
    then load the saved model artifact for live use.
    """
    ensure_training_files_exist()
    if not hasattr(uc, "main"):
        raise RuntimeError("Selected classifier module does not expose main().")

    print("Running classifier full pipeline to train/test model...")
    uc.main()

    candidate_paths = [
        path,
        DEFAULT_LIVE_MODEL_PATH,
        LEGACY_LIVE_MODEL_PATH,
        os.path.join(str(RESULTS_DIR / "manual"), "trained_updated_model.npz"),
        os.path.join(str(RESULTS_DIR / "manual"), "trained_manual_model.npz"),
    ]

    params = threshold = norm_mean = norm_std = None
    loaded_path = None
    for model_path in candidate_paths:
        params, threshold, norm_mean, norm_std = load_saved_model(model_path)
        if params is not None:
            loaded_path = model_path
            break

    if params is None:
        raise RuntimeError(
            "classifier completed but no valid saved model was found at "
            f"{path}"
        )

    print(f"Loaded freshly trained model: {loaded_path}")
    print(f"Using threshold: {threshold:.4f}")
    return params, threshold, norm_mean, norm_std


def get_model(force_retrain: bool):
    """Load cached model if possible; otherwise train and cache."""
    if not force_retrain:
        candidate_paths = [
            MODEL_PATH,
            DEFAULT_LIVE_MODEL_PATH,
            LEGACY_LIVE_MODEL_PATH,
            os.path.join(str(RESULTS_DIR / "manual"), "trained_updated_model.npz"),
            os.path.join(str(RESULTS_DIR / "manual"), "trained_manual_model.npz"),
        ]
        for model_path in candidate_paths:
            params, threshold, norm_mean, norm_std = load_saved_model(model_path)
            if params is not None:
                print(f"Loaded model cache: {model_path}")
                print(f"Using threshold: {threshold:.4f}")
                return params, threshold, norm_mean, norm_std

    return train_and_cache_model(MODEL_PATH)


def record_clip(seconds: int, sample_rate: int) -> np.ndarray:
    """Record mono audio clip from default microphone."""
    n_samples = int(seconds * sample_rate)
    print(f"\nRecording for {seconds} seconds... speak now.")
    audio = sd.rec(
        n_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete.")
    return audio.reshape(-1)


def prepare_live_clip(raw_audio: np.ndarray, target_samples: int,
                      apply_trim_silence: bool = True) -> np.ndarray:
    """
    Make live audio closer to training clips:
    - trim silence
    - ensure fixed length by crop/tile
    """
    trimmed = uc.trim_silence(raw_audio) if apply_trim_silence else raw_audio
    if len(trimmed) == 0:
        trimmed = raw_audio
    if len(trimmed) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if len(trimmed) >= target_samples:
        return trimmed[:target_samples].astype(np.float32)

    reps = int(np.ceil(target_samples / len(trimmed)))
    tiled = np.tile(trimmed, reps)[:target_samples]
    return tiled.astype(np.float32)


def classify_audio(
    raw_audio: np.ndarray,
    params: dict,
    threshold: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    target_samples: int,
    apply_trim_silence: bool = True,
    hidden_activation: str = DEFAULT_HIDDEN_ACTIVATION,
):
    """Classify one audio array and return (speaker, score, confidence, margin)."""
    prepared = prepare_live_clip(
        raw_audio,
        target_samples,
        apply_trim_silence=apply_trim_silence,
    )
    feat = uc.clip_to_spectrogram(prepared).reshape(1, -1)
    if hasattr(uc, "apply_normalizer"):
        feat = uc.apply_normalizer(feat, norm_mean, norm_std)
    if PREDICT_NEEDS_ACTIVATION:
        score = float(uc.predict(feat, params, hidden_activation)[0])
    else:
        score = float(uc.predict(feat, params)[0])

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
    """Classify one recorded clip and print prediction."""
    clip_duration = int(getattr(uc, "CLIP_DURATION", 5))
    sample_rate = int(getattr(uc, "SAMPLE_RATE", 22050))
    speaker, score, confidence, margin = classify_audio(
        raw_audio,
        params,
        threshold,
        norm_mean,
        norm_std,
        int(clip_duration * sample_rate),
        apply_trim_silence=True,
        hidden_activation=hidden_activation,
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
    window_seconds: int,
    hop_seconds: float,
    min_rms: float,
    min_margin: float,
    smooth_votes: int,
    print_all: bool,
    live_trim_silence: bool,
    input_latency: str,
    infer_every: int,
    hidden_activation: str,
):
    """
    Continuously read microphone audio and classify using a rolling window.
    """
    sample_rate = int(getattr(uc, "SAMPLE_RATE", 22050))
    window_samples = int(window_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)

    if window_samples <= 0:
        raise ValueError("window_seconds must be > 0")
    if hop_samples <= 0:
        raise ValueError("hop_seconds must be > 0")
    if hop_samples > window_samples:
        raise ValueError("hop_seconds must be <= window_seconds")
    if infer_every <= 0:
        raise ValueError("infer_every must be >= 1")

    # Audio callback writes into a lock-protected ring buffer.
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
        f"Infer every: {infer_every} tick(s)"
    )
    print("Press Ctrl+C to stop.\n")

    # Warm up feature extraction/inference once before real-time loop
    # to avoid one-time startup stalls that can trigger an overflow warning.
    _ = classify_audio(
        np.zeros(window_samples, dtype=np.float32),
        params,
        threshold,
        norm_mean,
        norm_std,
        window_samples,
        apply_trim_silence=False,
        hidden_activation=hidden_activation,
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

            return np.concatenate(
                (ring_buffer[start:], ring_buffer[:end]),
                axis=0,
            ).astype(np.float32, copy=False)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=0,
        latency=input_latency,
        callback=audio_callback,
    ) as stream:
        _ = stream
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
                    "Try increasing --hop-seconds (e.g., 2.0+), keep --input-latency high, "
                    "and close CPU-heavy apps."
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
                apply_trim_silence=live_trim_silence,
                hidden_activation=hidden_activation,
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
    default_clip_duration = int(getattr(uc, "CLIP_DURATION", 5))
    default_sample_rate = int(getattr(uc, "SAMPLE_RATE", 22050))

    parser = argparse.ArgumentParser(description="Live mic speaker test (Gabriel vs Raiz).")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining from source recordings before mic testing.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Record and classify one clip, then exit.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=default_clip_duration,
        help=f"One-shot recording length for --once (default: {default_clip_duration}).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=default_clip_duration,
        help=f"Rolling window size for continuous mode (default: {default_clip_duration}).",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=1.0,
        help="Continuous mode update interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=0.005,
        help="Ignore low-energy audio below this RMS in continuous mode (default: 0.005).",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.03,
        help="Mark prediction as Uncertain if |score-threshold| is below this margin (default: 0.03).",
    )
    parser.add_argument(
        "--smooth-votes",
        type=int,
        default=5,
        help="Majority vote window size for stable live labels (default: 5).",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every update (default prints only when speaker label changes).",
    )
    parser.add_argument(
        "--live-trim-silence",
        action="store_true",
        help=(
            "Apply silence trimming on each live window. This is slower and may cause "
            "mic overflow on some systems; default is OFF for better realtime stability."
        ),
    )
    parser.add_argument(
        "--input-latency",
        choices=["low", "high"],
        default="high",
        help=(
            "Audio input latency hint for sounddevice (default: high). "
            "Use high to reduce overflow risk; low reduces capture latency."
        ),
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help=(
            "Run model inference every N update ticks (default: 1). "
            "Use 2 or 3 to reduce CPU load and improve realtime stability."
        ),
    )
    parser.add_argument(
        "--hidden-activation",
        type=str,
        default=DEFAULT_HIDDEN_ACTIVATION,
        help=(
            "Hidden-layer activation used by classifier modules whose predict() requires it "
            f"(default: {DEFAULT_HIDDEN_ACTIVATION})."
        ),
    )
    args = parser.parse_args()

    if PREDICT_NEEDS_ACTIVATION and MODULE_ACTIVATIONS and args.hidden_activation not in MODULE_ACTIVATIONS:
        print(
            f"ERROR: --hidden-activation must be one of: {', '.join(MODULE_ACTIVATIONS)} "
            f"(got: {args.hidden_activation})"
        )
        sys.exit(1)

    try:
        params, threshold, norm_mean, norm_std = get_model(force_retrain=args.retrain)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    try:
        if args.once:
            raw_audio = record_clip(args.seconds, default_sample_rate)
            classify_clip(
                raw_audio,
                params,
                threshold,
                norm_mean,
                norm_std,
                hidden_activation=args.hidden_activation,
            )
            return

        run_live_stream(
            params=params,
            threshold=threshold,
            norm_mean=norm_mean,
            norm_std=norm_std,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            min_rms=args.min_rms,
            min_margin=args.min_margin,
            smooth_votes=args.smooth_votes,
            print_all=args.print_all,
            live_trim_silence=args.live_trim_silence,
            input_latency=args.input_latency,
            infer_every=args.infer_every,
            hidden_activation=args.hidden_activation,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

"""
Live microphone speaker test for the updated voice classifier.

Usage examples:
  python live_speaker_test.py
  python live_speaker_test.py --retrain
  python live_speaker_test.py --once
  python live_speaker_test.py --window-seconds 5 --hop-seconds 1
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: Missing dependency 'sounddevice'.")
    print("Install it with: pip install sounddevice")
    sys.exit(1)

import updated_classifier as uc


MODEL_PATH = uc.MODEL_PATH


def load_saved_model(path: str):
    """Load model params, threshold, and normalizer stats from disk."""
    return uc.load_saved_model(path)


def ensure_training_files_exist():
    """Ensure the four source files required by updated_classifier are present."""
    required = [
        uc.GABRIEL_TRAIN_FILE,
        uc.RAIZ_TRAIN_FILE,
        uc.GABRIEL_TEST_FILE,
        uc.RAIZ_TEST_FILE,
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
    print("Running updated_classifier full pipeline to train/test model...")
    uc.main()

    params, threshold, norm_mean, norm_std = load_saved_model(path)
    if params is None:
        raise RuntimeError(
            "updated_classifier completed but no valid saved model was found at "
            f"{path}"
        )

    print(f"Loaded freshly trained model: {path}")
    print(f"Using threshold: {threshold:.4f}")
    return params, threshold, norm_mean, norm_std


def get_model(force_retrain: bool):
    """Load cached model if possible; otherwise train and cache."""
    if not force_retrain:
        params, threshold, norm_mean, norm_std = load_saved_model(MODEL_PATH)
        if params is not None:
            print(f"Loaded model cache: {MODEL_PATH}")
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


def prepare_live_clip(raw_audio: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Make live audio closer to training clips:
    - trim silence
    - ensure fixed length by crop/tile
    """
    trimmed = uc.trim_silence(raw_audio)
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
):
    """Classify one audio array and return (speaker, score, confidence, margin)."""
    prepared = prepare_live_clip(raw_audio, target_samples)
    feat = uc.clip_to_spectrogram(prepared).reshape(1, -1)
    feat = uc.apply_normalizer(feat, norm_mean, norm_std)
    score = float(uc.predict(feat, params)[0])

    pred = 1 if score >= threshold else 0
    speaker = "Raiz" if pred == 1 else "Gabriel"
    confidence = score if pred == 1 else (1.0 - score)
    margin = abs(score - threshold)

    return speaker, score, confidence, margin


def classify_clip(raw_audio: np.ndarray, params: dict, threshold: float,
                  norm_mean: np.ndarray, norm_std: np.ndarray):
    """Classify one recorded clip and print prediction."""
    speaker, score, confidence, margin = classify_audio(
        raw_audio,
        params,
        threshold,
        norm_mean,
        norm_std,
        int(uc.CLIP_DURATION * uc.SAMPLE_RATE),
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
):
    """
    Continuously read microphone audio and classify using a rolling window.
    """
    sample_rate = uc.SAMPLE_RATE
    window_samples = int(window_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)

    if window_samples <= 0:
        raise ValueError("window_seconds must be > 0")
    if hop_samples <= 0:
        raise ValueError("hop_seconds must be > 0")
    if hop_samples > window_samples:
        raise ValueError("hop_seconds must be <= window_seconds")

    buffer = np.zeros(0, dtype=np.float32)
    last_label = None
    overflow_warned = False
    vote_buffer = deque(maxlen=max(1, smooth_votes))

    print("\nContinuous live classifier ready.")
    print(
        f"Window: {window_seconds}s | Update every: {hop_seconds:.2f}s | "
        f"Min RMS: {min_rms:.4f} | Min margin: {min_margin:.4f} | "
        f"Smoothing: {max(1, smooth_votes)}"
    )
    print("Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=hop_samples,
    ) as stream:
        while True:
            frames, overflowed = stream.read(hop_samples)
            if overflowed and not overflow_warned:
                print("WARNING: Audio input overflow detected. Predictions may be delayed.")
                overflow_warned = True

            chunk = frames[:, 0]
            buffer = np.concatenate([buffer, chunk], axis=0)

            if len(buffer) < window_samples:
                continue
            if len(buffer) > window_samples:
                buffer = buffer[-window_samples:]

            rms = float(np.sqrt(np.mean(buffer ** 2)))
            if rms < min_rms:
                if print_all:
                    print("Prediction: (silence/noise) | waiting for clearer speech")
                continue

            speaker, score, confidence, margin = classify_audio(
                buffer,
                params,
                threshold,
                norm_mean,
                norm_std,
                window_samples,
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
        default=uc.CLIP_DURATION,
        help=f"One-shot recording length for --once (default: {uc.CLIP_DURATION}).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=uc.CLIP_DURATION,
        help=f"Rolling window size for continuous mode (default: {uc.CLIP_DURATION}).",
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
    args = parser.parse_args()

    try:
        params, threshold, norm_mean, norm_std = get_model(force_retrain=args.retrain)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    try:
        if args.once:
            raw_audio = record_clip(args.seconds, uc.SAMPLE_RATE)
            classify_clip(raw_audio, params, threshold, norm_mean, norm_std)
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
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

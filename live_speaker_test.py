"""
Live microphone speaker test for the manual voice classifier.

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

import manual_classifier as mc


MODEL_PATH = os.path.join(mc.OUTPUT_DIR, "trained_manual_model.npz")


def load_saved_model(path: str):
    """Load model params and threshold from disk if available."""
    if not os.path.exists(path):
        return None, None

    data = np.load(path)
    required = {"W1", "b1", "W2", "b2", "threshold"}
    if not required.issubset(set(data.files)):
        return None, None

    params = {
        "W1": data["W1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "W2": data["W2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
    }
    threshold = float(data["threshold"])
    return params, threshold


def train_and_cache_model(path: str):
    """Train using existing recordings and cache model params + threshold."""
    print("Training model from recordings...")

    y_gabriel = mc.trim_silence(mc.load_and_convert_m4a(mc.GABRIEL_FILE))
    gabriel_clips = mc.segment_audio(y_gabriel, mc.GABRIEL_LABEL, "Gabriel")

    y_raiz = mc.trim_silence(mc.load_and_convert_m4a(mc.RAIZ_FILE))
    raiz_clips = mc.segment_audio(y_raiz, mc.RAIZ_LABEL, "Raiz")

    if len(gabriel_clips) == 0 or len(raiz_clips) == 0:
        raise RuntimeError("One or both speakers produced 0 clips. Check source audio files.")

    X_train, _, y_train, _ = mc.build_dataset(gabriel_clips, raiz_clips)
    params, _ = mc.train(X_train, y_train)

    train_scores = mc.predict(X_train, params)
    threshold = mc.select_threshold_by_youden(y_train, train_scores)

    np.savez(
        path,
        W1=params["W1"],
        b1=params["b1"],
        W2=params["W2"],
        b2=params["b2"],
        threshold=np.array(threshold, dtype=np.float32),
    )
    print(f"Saved model cache: {path}")
    print(f"Using threshold: {threshold:.4f}")
    return params, threshold


def get_model(force_retrain: bool):
    """Load cached model if possible; otherwise train and cache."""
    if not force_retrain:
        params, threshold = load_saved_model(MODEL_PATH)
        if params is not None:
            print(f"Loaded model cache: {MODEL_PATH}")
            print(f"Using threshold: {threshold:.4f}")
            return params, threshold

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
    trimmed = mc.trim_silence(raw_audio)
    if len(trimmed) == 0:
        trimmed = raw_audio
    if len(trimmed) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if len(trimmed) >= target_samples:
        return trimmed[:target_samples].astype(np.float32)

    reps = int(np.ceil(target_samples / len(trimmed)))
    tiled = np.tile(trimmed, reps)[:target_samples]
    return tiled.astype(np.float32)


def classify_audio(raw_audio: np.ndarray, params: dict, threshold: float, target_samples: int):
    """Classify one audio array and return (speaker, score, confidence, margin)."""
    prepared = prepare_live_clip(raw_audio, target_samples)
    feat = mc.clip_to_spectrogram(prepared).reshape(1, -1)
    score = float(mc.predict(feat, params)[0])

    pred = 1 if score >= threshold else 0
    speaker = "Raiz" if pred == 1 else "Gabriel"
    confidence = score if pred == 1 else (1.0 - score)
    margin = abs(score - threshold)

    return speaker, score, confidence, margin


def classify_clip(raw_audio: np.ndarray, params: dict, threshold: float):
    """Classify one recorded clip and print prediction."""
    speaker, score, confidence, margin = classify_audio(
        raw_audio,
        params,
        threshold,
        int(mc.CLIP_DURATION * mc.SAMPLE_RATE),
    )
    print(f"Prediction: {speaker}")
    print(
        f"Score(Raiz=1): {score:.4f} | Threshold: {threshold:.4f} | "
        f"Confidence: {confidence:.4f} | Margin: {margin:.4f}"
    )


def run_live_stream(
    params: dict,
    threshold: float,
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
    sample_rate = mc.SAMPLE_RATE
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
        default=mc.CLIP_DURATION,
        help=f"One-shot recording length for --once (default: {mc.CLIP_DURATION}).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=mc.CLIP_DURATION,
        help=f"Rolling window size for continuous mode (default: {mc.CLIP_DURATION}).",
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

    for fname in [mc.GABRIEL_FILE, mc.RAIZ_FILE]:
        if not os.path.exists(fname):
            print(f"ERROR: Missing required training file: {fname}")
            sys.exit(1)

    params, threshold = get_model(force_retrain=args.retrain)

    try:
        if args.once:
            raw_audio = record_clip(args.seconds, mc.SAMPLE_RATE)
            classify_clip(raw_audio, params, threshold)
            return

        run_live_stream(
            params=params,
            threshold=threshold,
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

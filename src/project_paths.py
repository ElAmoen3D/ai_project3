from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
RESULTS_DIR = PROJECT_ROOT / "results"

GABRIEL_TRAIN_FILE = str(AUDIO_DIR / "gabriel_samples.m4a")
RAIZ_TRAIN_FILE = str(AUDIO_DIR / "raiz_samples.m4a")
GABRIEL_TEST_FILE = str(AUDIO_DIR / "gabriel_test.m4a")
RAIZ_TEST_FILE = str(AUDIO_DIR / "raiz_test.m4a")


def ensure_base_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

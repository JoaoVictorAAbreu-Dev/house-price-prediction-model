"""Load the local model only with the dependencies used to train it."""

import json
import warnings
from pathlib import Path

import joblib
import numpy
import scipy
import sklearn
from sklearn.exceptions import InconsistentVersionWarning


class ModelLoadError(Exception):
    """The saved model is missing or cannot be used in this environment."""


def load_trained_model(model_path, metadata_path):
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if not model_path.is_file() or not metadata_path.is_file():
        raise ModelLoadError("Model files are missing. Run `python train.py` first.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelLoadError("Model metadata is unreadable. Run `python train.py` again.") from exc

    current = {
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }
    if not isinstance(metadata, dict) or not isinstance(metadata.get("training_environment", {}), dict):
        raise ModelLoadError("Model metadata is invalid. Run `python train.py` again.")
    trained_with = metadata.get("training_environment", {})
    mismatches = [
        f"{name}: trained with {version}, installed {current[name]}"
        for name, version in trained_with.items()
        if name in current and version != current[name]
    ]
    if mismatches:
        raise ModelLoadError(
            "Model dependency mismatch (" + "; ".join(mismatches)
            + "). Install requirements.txt or run `python train.py` again."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            return joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(
            "Saved model is incompatible or unreadable. Run `python train.py` again."
        ) from exc

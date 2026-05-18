from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics_json(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_metrics = _json_safe(metrics)

    with open(path, "w") as f:
        json.dump(clean_metrics, f, indent=2)


def save_table_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_array_npy(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def save_figure(fig, path: str | Path, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def make_run_dir(
    base_dir: str | Path,
    run_name: str,
    add_timestamp: bool = True,
) -> Path:
    base_dir = Path(base_dir)

    if add_timestamp:
        run_dir = base_dir / f"{timestamp()}_{run_name}"
    else:
        run_dir = base_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_safe(obj):
    """
    Convert numpy/pandas objects to JSON-safe Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_)):
        return bool(obj)

    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple, np.ndarray)) else False:
        return None

    return obj
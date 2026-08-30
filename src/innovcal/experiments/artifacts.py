import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_table(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_array_npz(
    path: str | Path,
    **arrays,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_array_npz(
    path: str | Path,
) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def save_json(
    obj: dict,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def save_current_figure(
    path: str | Path,
    dpi: int = 300,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def maybe_save_table(
    df: pd.DataFrame,
    path: str | Path | None,
) -> None:
    if path is not None:
        save_table(df, path)


def maybe_save_npz(
    path: str | Path | None,
    **arrays,
) -> None:
    if path is not None:
        save_array_npz(path, **arrays)
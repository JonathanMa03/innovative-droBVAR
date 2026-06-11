from pathlib import Path
from datetime import datetime


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_output_dir(
    base: str | Path,
    name: str,
    test: bool = False,
    timestamp: bool = False,
) -> Path:
    base = Path(base)

    if test:
        base = base / "test"

    if timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{name}"

    out = base / name
    out.mkdir(parents=True, exist_ok=True)

    return out


def result_dirs(
    experiment_name: str,
    test: bool = False,
) -> dict[str, Path]:
    root = project_root()

    dirs = {
        "results": make_output_dir(root / "results", experiment_name, test=test),
        "tables": make_output_dir(root / "results" / "tables", experiment_name, test=test),
        "figures": make_output_dir(root / "results" / "figures", experiment_name, test=test),
        "models": make_output_dir(root / "results" / "models", experiment_name, test=test),
        "forecasts": make_output_dir(root / "results" / "forecasts", experiment_name, test=test),
        "innovations": make_output_dir(root / "results" / "innovations", experiment_name, test=test),
        "residuals": make_output_dir(root / "results" / "residuals", experiment_name, test=test),
        "logs": make_output_dir(root / "logs" / "experiments", experiment_name, test=test),
    }

    return dirs
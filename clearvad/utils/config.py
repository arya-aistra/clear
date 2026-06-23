"""Config + reproducibility helpers.

All ClearVAD config lives in YAML (GSD reproducibility contract). Seeds are set
globally and recorded. torch/numpy seeding is best-effort (torch optional).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import yaml

PathLike = Union[str, os.PathLike]


class _RobustDumper(yaml.SafeDumper):
    """SafeDumper that stringifies anything it can't otherwise represent.

    Needed because values like ``torch.__version__`` are ``str`` *subclasses*
    (TorchVersion) or numpy scalars, which SafeDumper has no representer for and would
    otherwise raise RepresenterError on. Unknown objects are emitted as plain strings.
    """


def _represent_fallback(dumper: yaml.Dumper, data: Any):
    return dumper.represent_str(str(data))


# Catch-all for unknown types (yaml_representers[None] == represent_undefined by default).
_RobustDumper.add_representer(None, _represent_fallback)


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def save_yaml(data: Dict[str, Any], path: PathLike) -> None:
    """Write a dict to YAML, creating parent dirs. Unknown objects are stringified."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, Dumper=_RobustDumper, sort_keys=False,
                  default_flow_style=False)


def set_global_seed(seed: int = 1234, deterministic: bool = True) -> int:
    """Seed python, numpy, and (if available) torch. Returns the seed used.

    With ``deterministic=True`` we also pin cuDNN to deterministic mode so that
    distillation runs are reproducible across the GPU server.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
    return seed

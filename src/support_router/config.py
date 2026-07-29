"""Central constants and paths.

Kept deliberately small: experiment settings live in `conf/params.yaml` and are loaded
through `load_params()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "train.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
PARAMS_PATH = PROJECT_ROOT / "conf" / "params.yaml"

#: The four routes. Order is fixed and used for every confusion matrix / report column,
#: so it must not be sorted or derived from the data (a class absent from a fold would
#: otherwise silently shift the columns).
LABELS: tuple[str, ...] = (
    "account-access",
    "transaction-dispute",
    "fraud-report",
    "general",
)

#: The route where a false negative is most expensive: a missed fraud report sits in a
#: general queue while an account is actively being drained. Drives metric selection and
#: the CI promotion gate.
CRITICAL_LABEL = "fraud-report"

TEXT_COL = "text"
LABEL_COL = "label"

RANDOM_SEED = 20260728


@dataclass(frozen=True)
class Params:
    """Typed view over `conf/params.yaml`."""

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def cv(self) -> dict[str, Any]:
        return self.raw["cv"]

    @property
    def grouping(self) -> dict[str, Any]:
        return self.raw["grouping"]

    @property
    def promotion(self) -> dict[str, Any]:
        return self.raw["promotion"]

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw["llm"]


def load_params(path: Path | str | None = None) -> Params:
    with open(path or PARAMS_PATH) as fh:
        return Params(yaml.safe_load(fh))

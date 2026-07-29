"""Shared fixtures.

The synthetic corpus deliberately mimics the real generator: a handful of templates,
each rendered several times with a different opener, closer and asset. Tests that assert
things about leakage need data with that structure, and building it explicitly makes the
property being tested visible rather than dependent on the real file.
"""

from __future__ import annotations

import pandas as pd
import pytest

OPENERS = ["Hi, ", "Hey, ", "Urgent: ", "Please help. ", ""]
CLOSERS = [" Thanks.", " Please advise.", " Appreciate any help.", ""]
ASSETS = ["BTC", "ETH", "SOL", "Cardano", "Polygon"]

# Four templates per label. Fewer than that and a grouped 4-fold split can strand an
# entire class in one test fold, leaving the training side with no examples of it - the
# fold then scores 0 for reasons unrelated to the model. `make_splits` warns about this;
# the fixture is sized to avoid provoking it.
TEMPLATES: list[tuple[str, str]] = [
    ("I can't log into my account, it keeps saying my password is wrong", "account-access"),
    ("The app won't let me sign in, it just spins on the login screen", "account-access"),
    ("My login has been blocked for security reasons and I need access", "account-access"),
    ("Two factor authentication codes never arrive on my phone", "account-access"),
    ("My withdrawal of {n} {asset} shows completed but I never received it", "transaction-dispute"),
    ("I cancelled my buy order but {n} {asset} was still taken", "transaction-dispute"),
    ("My limit order should not have executed at that price, please review",
     "transaction-dispute"),
    ("A transaction has been stuck in pending for two days and the funds are gone",
     "transaction-dispute"),
    ("Someone withdrew {n} {asset} from my account that I never authorized", "fraud-report"),
    ("I got a phishing email pretending to be support and entered my credentials",
     "fraud-report"),
    ("My funds are gone, {n} {asset} disappeared overnight and I believe this is fraud",
     "fraud-report"),
    ("Someone else has taken over my account and changed my email", "fraud-report"),
    ("How long do {asset} withdrawals usually take to process?", "general"),
    ("What's the minimum amount I can buy of {asset}?", "general"),
    ("Where can I download my tax documents for last year?", "general"),
    ("How are transaction fees calculated on your platform?", "general"),
]


def _render(template: str, i: int) -> str:
    text = template.format(n=round(0.5 + i * 0.7, 1), asset=ASSETS[i % len(ASSETS)])
    return f"{OPENERS[i % len(OPENERS)]}{text}{CLOSERS[i % len(CLOSERS)]}"


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """Template-generated corpus: 16 templates x 5 renderings = 80 rows."""
    rows = []
    for template, label in TEMPLATES:
        for i in range(5):
            rows.append({"text": _render(template, i), "label": label})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def real_frame():
    """The checked-in training data."""
    from support_router.data import DataValidationError, load_training_data

    try:
        return load_training_data()
    except DataValidationError as exc:
        pytest.skip(f"training data unavailable: {exc}")


@pytest.fixture
def trained_model_dir(tmp_path, synthetic_frame):
    """Train a fast model into a temp directory and point the predict cache at it."""
    from support_router import inference as predict_module
    from support_router.train import train

    data_path = tmp_path / "train.csv"
    synthetic_frame.to_csv(data_path, index=False)

    out_dir = tmp_path / "artifacts"
    train(
        model="logistic_regression",
        out_dir=out_dir,
        data_path=data_path,
        evaluate=False,   # the point of the fixture is the artifact, not its score
        track=False,
    )
    predict_module.reset_cache()
    yield out_dir
    predict_module.reset_cache()

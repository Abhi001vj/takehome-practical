"""HTTP contract tests.


`TestClient` runs the lifespan handler, so the startup-load path is covered too.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the `api` extra")

from fastapi.testclient import TestClient  # noqa: E402

from support_router import inference as predict_module  # noqa: E402
from support_router.api import service as api_module  # noqa: E402
from support_router.config import LABELS  # noqa: E402


@pytest.fixture
def client(trained_model_dir, monkeypatch):
    """A client whose app is pointed at a freshly trained temp artifact."""
    monkeypatch.setenv(api_module.MODEL_DIR_ENV, str(trained_model_dir))
    predict_module.reset_cache()
    with TestClient(api_module.create_app()) as c:
        yield c
    predict_module.reset_cache()


@pytest.fixture
def modelless_client(tmp_path, monkeypatch):
    """A client started with no artifact on disk - the cold-deploy case."""
    monkeypatch.setenv(api_module.MODEL_DIR_ENV, str(tmp_path / "empty"))
    predict_module.reset_cache()
    with TestClient(api_module.create_app()) as c:
        yield c
    predict_module.reset_cache()


class TestHealth:
    def test_reports_ready_when_a_model_is_loaded(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_starts_without_a_model_instead_of_crashing(self, modelless_client):
        """A cold container must come up so the trainer can populate the volume.

        If startup raised instead, `docker compose up` could never reach a state where
        the API is running and waiting for its first artifact.
        """
        response = modelless_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"
        assert "train" in (body["detail"] or "")

    def test_inference_is_503_not_500_without_a_model(self, modelless_client):
        response = modelless_client.post("/predict", json={"text": "I cannot log in"})
        assert response.status_code == 503


class TestPredict:
    def test_returns_a_known_route(self, client):
        response = client.post("/predict", json={"text": "I can't log into my account"})
        assert response.status_code == 200
        assert response.json()["label"] in LABELS

    def test_confidence_is_a_probability_over_the_known_labels(self, client):
        body = client.post("/predict", json={"text": "Someone drained my wallet"}).json()
        assert 0.0 <= body["confidence"] <= 1.0
        assert set(body["scores"]) == set(LABELS)
        assert body["scores"][body["label"]] == pytest.approx(body["confidence"])
        assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_blank_text_is_a_client_error(self, client):
        """400, not 500: the caller sent something they can fix."""
        assert client.post("/predict", json={"text": "   "}).status_code == 400

    def test_missing_field_is_a_422(self, client):
        assert client.post("/predict", json={}).status_code == 422


class TestBatch:
    def test_positions_align_with_the_input(self, client):
        texts = ["I can't log in", "How do fees work?", "Someone stole my funds"]
        body = client.post("/predict/batch", json={"texts": texts}).json()
        assert len(body["predictions"]) == len(texts)
        assert body["n_invalid"] == 0

    def test_batch_agrees_with_single_calls(self, client):
        """Batching must not change the answer, only the cost of getting it."""
        texts = ["I can't log into my account", "What are the withdrawal limits?"]
        batched = client.post("/predict/batch", json={"texts": texts}).json()["predictions"]
        singles = [client.post("/predict", json={"text": t}).json() for t in texts]
        assert [b["label"] for b in batched] == [s["label"] for s in singles]

    def test_skip_invalid_holds_the_position_open(self, client):
        """A null must appear *at the bad row's index* so callers can zip results back."""
        texts = ["I can't log in", "", "Someone stole my funds"]
        body = client.post(
            "/predict/batch", json={"texts": texts, "skip_invalid": True}
        ).json()
        assert body["predictions"][1] is None
        assert body["n_invalid"] == 1
        assert body["predictions"][0] is not None and body["predictions"][2] is not None

    def test_a_bad_row_fails_the_request_by_default(self, client):
        response = client.post("/predict/batch", json={"texts": ["fine", "  "]})
        assert response.status_code == 400

    def test_empty_batch_is_rejected_by_the_schema(self, client):
        assert client.post("/predict/batch", json={"texts": []}).status_code == 422


class TestInfo:
    def test_exposes_what_is_deployed(self, client):
        body = client.get("/info").json()
        assert body["labels"] == list(LABELS)
        assert body["model_name"] == "logistic_regression"
        assert body["trained_at"]


class TestPackageExports:
    """Pin the package-level `predict(text) -> label` API.

    This broke twice while being written - first the lazy `__getattr__` recursed, then a
    submodule named `predict` shadowed the function it was supposed to export. Both
    failures are invisible until someone calls it, so they are pinned here.
    """

    def test_predict_is_importable_and_callable_from_the_package(self):
        import support_router

        assert callable(support_router.predict)
        assert callable(support_router.predict_batch)

    def test_no_submodule_shadows_the_exported_function(self):
        """Importing `support_router.inference` must not rebind `support_router.predict`.

        This is exactly what a module named `support_router.predict` used to do.
        """
        import support_router
        import support_router.inference  # noqa: F401

        assert callable(support_router.predict)

    def test_unknown_attributes_still_raise(self):
        import support_router

        with pytest.raises(AttributeError):
            support_router.does_not_exist

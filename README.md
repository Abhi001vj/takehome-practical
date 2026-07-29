# Support-ticket router

A four-class text classifier for crypto/fintech support messages. It routes each message
to `account-access`, `transaction-dispute`, `fraud-report`, or `general` and provides a
Python interface, CLI, batch scorer, and optional FastAPI service.

The central modeling decision is the evaluation split. The 400 messages are generated
from lightly varied templates: they collapse to 80 near-duplicate groups, and a normal
row-level split puts a template sibling in training for 95.8% of validation rows. Model
selection therefore uses repeated stratified group cross-validation. The naive split is
retained only to make the leakage gap visible.

## Exercise requirements and evidence

| requirement | implementation | how to verify |
|---|---|---|
| Four-route classifier | Classical TF-IDF baselines, linear models, Naive Bayes, trees, frozen-embedding heads, and direct Qwen comparisons all implement the same four-label interface. | `uv run support-router cv --models logistic_regression --schemes grouped` |
| Proper evaluation | Five-fold stratified group CV, repeated four times, keeps all members of a near-duplicate template family in one fold. Row-level CV is reported only as a leakage diagnostic. | `uv run support-router leakage`; see [reports/comparison.md](reports/comparison.md) |
| Imbalance decision and justified metric | Macro-F1 is the selection metric, `fraud-report` recall is a separate guardrail, and supported linear models use balanced class weights. Text resampling was deliberately avoided. | See [Metric and imbalance policy](#metric-and-imbalance-policy) and [reports/REPORT.md](reports/REPORT.md) |
| Clean, tested Python and validation | Shared validation rejects missing, non-string, too-short, and oversized messages. The same inference implementation is used by Python, CLI, batch scoring, and API. | `uv run ruff check .`; `uv run pytest -q` (107 tests) |
| `predict(text) -> label` | `support_router.inference.predict` and the `support-router predict` CLI load the persisted artifact. | See [Inference model selection](#inference-model-selection) |
| Holdout scoring | The scorer reads a CSV, preserves row order, writes predictions, optionally writes confidence, and separates invalid rows. | `uv run support-router score holdout.csv --output predictions.csv --confidence` |
| Optional LLM comparison | Qwen 2.5 1.5B zero-shot and few-shot routes are evaluated through an OpenAI-compatible endpoint with output validation and parse-failure accounting. | See [Direct Qwen comparison](#direct-qwen-comparison) |
| Optional API | FastAPI provides single and batch prediction, health, model information, validation, and OpenAPI documentation. | See [Run and test the API](#run-and-test-the-api) |
| Optional containerization and CI | Compose packages training/API serving and optional NVIDIA vLLM; GitHub Actions runs lint, tests, evaluation, a promotion gate, and uploads reports. | See [Containers and CI](#containers-and-ci) |
| Scope and trade-offs | Priorities, exclusions, next steps, and the honest time spent are stated explicitly. | See [Scope and trade-offs](#scope-and-trade-offs) |

## Results

The selected model is a frozen MiniLM sentence encoder with a class-weighted logistic
regression head.

| model | grouped macro-F1 | std | `fraud-report` recall | naive macro-F1 |
|---|---:|---:|---:|---:|
| `embedding_logreg` | **0.971** | 0.038 | **0.945** | 0.998 |
| `embedding_lightgbm` | 0.889 | 0.059 | 0.765 | 0.978 |
| `linear_svc` | 0.829 | 0.090 | 0.810 | 0.996 |
| `logistic_regression` | 0.808 | 0.088 | 0.780 | 0.997 |
| Qwen 2.5 1.5B zero-shot | 0.762 | 0.086 | 0.840 | 0.769 |
| Qwen 2.5 1.5B few-shot | 0.706 | 0.096 | 0.925 | 0.701 |
| `most_frequent` | 0.143 | 0.002 | 0.000 | 0.143 |

The complete results, confusion matrices, EDA, timing, MLflow inventory, and deployment
trade-offs are in [reports/REPORT.md](reports/REPORT.md). The machine-readable comparison
is [reports/comparison.json](reports/comparison.json).

### Metric and imbalance policy

Macro-F1 is the primary metric because all four queues matter and the row distribution is
3.2:1. Accuracy alone is unsafe: predicting `general` for every message reaches 40%
accuracy while recalling no fraud reports. Macro-F1 is paired with a separately reported
`fraud-report` recall and an 0.80 promotion floor, because one averaged metric cannot
fully represent the asymmetric cost of missing fraud.

Linear models use balanced class weights. Resampling was deliberately avoided: the
template-level imbalance is only 1.6:1, so synthesizing or duplicating rows would mainly
amplify the templating rather than add information. Per-class recall/F1, confusion
matrices, fold variance, and the promotion guardrail expose harm hidden by aggregate
metrics.

## Quick start

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --all-extras
uv run support-router test
uv run support-router experiment \
  --models all,embedding_logreg,embedding_lightgbm \
  --track \
  --register-winner
```

The experiment regenerates the EDA, runs the selected models under grouped and naive CV,
writes comparison artifacts, refits the grouped-CV winner, benchmarks inference, runs the
promotion gate, and generates the consolidated report. The frozen encoder downloads once
from Hugging Face; use `logistic_regression` for the fully self-contained classical path.
Detailed stage-by-stage commands are in [COMMANDS.md](COMMANDS.md).

## Prediction and holdout interfaces

### Inference model selection

The default inference model is the artifact at `artifacts/model.joblib`. The committed
results select `embedding_logreg`: frozen
`sentence-transformers/all-MiniLM-L6-v2` embeddings followed by a class-weighted logistic
regression head. Confirm exactly what will be served with:

```bash
uv run support-router info
```

Training without `--out` replaces the default artifact. To compare or serve another model
without overwriting it, train into a named directory and pass that directory explicitly:

```bash
# Default selected model
uv run support-router train --model embedding_logreg

# Alternative classical model in its own artifact directory
uv run support-router train \
  --model logistic_regression \
  --out artifacts/logistic_regression

uv run support-router info --model-path artifacts/logistic_regression
uv run support-router predict \
  "I cannot access my account" \
  --model-path artifacts/logistic_regression \
  --scores
uv run support-router score holdout.csv \
  --output predictions-logistic.csv \
  --model-path artifacts/logistic_regression \
  --confidence
```

The MLflow `champion` alias records the reviewed release decision, but local inference does
not silently download that alias. Deployment selects a concrete, tested artifact directory,
which prevents the model changing underneath a running service.

### Single-message prediction

Train the chosen model and classify one message:

```bash
uv run support-router train --model embedding_logreg
uv run support-router predict \
  "Someone transferred ETH without my permission" \
  --scores
```

The Python interface is intentionally small:

```python
from support_router.inference import predict

label = predict("I cannot access my account after changing phones")
```

The return value is the label by default. Pass `with_scores=True` to receive the label,
confidence, and per-class scores when the selected estimator exposes probabilities.

### Holdout CSV scoring

The holdout-scoring entry point preserves input order and writes one prediction per valid
row:

```bash
uv run support-router score holdout.csv \
  --output predictions.csv \
  --confidence
```

The input must contain a `text` column (or use `--text-column`). If a `label` column is
present, the command additionally prints evaluation metrics; scoring `data/raw/train.csv`
therefore produces an in-sample sanity check, not a holdout estimate. The grouped-CV result
above is the honest model-selection evidence.

### Run and test the API

Start the service after training an artifact:

```bash
uv run support-router train --model embedding_logreg
uv run support-router serve
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/info
curl -fsS \
  -H 'Content-Type: application/json' \
  -d '{"text":"Someone transferred ETH without my permission"}' \
  http://127.0.0.1:8000/predict
```

The interactive OpenAPI UI is at <http://127.0.0.1:8000/docs>. To serve a different
artifact without changing code:

```bash
SUPPORT_ROUTER_MODEL_DIR=artifacts/logistic_regression \
  uv run support-router serve --port 8001
```

Run the API-specific tests with `uv run pytest -q tests/test_api.py`.

## Local MLflow tracking

MLflow runs locally; Docker is not required. Start the server in a separate terminal:

```bash
uv run mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --default-artifact-root "$PWD/mlruns" \
  --host 127.0.0.1 \
  --port 5001
```

Then set the tracking URI before running an experiment:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
uv run support-router cv --models classical --schemes grouped,naive --track
```

Open <http://127.0.0.1:5001>, switch to **Model training**, and select the
`support-routing` experiment. Each run records parameters, aggregate and per-class
metrics, fold details, confusion matrices, dataset digest/source, feature specification,
git state, timing, and model artifacts. Generative calls also produce MLflow traces.

The registry uses one model name, `support-router`, with independently inspectable
versions. Registration does not silently promote a model. After the grouped metrics and
gate have been reviewed, move the alias explicitly with
`uv run support-router promote --version <VERSION>`.

Local tracking state (`mlflow.db` and `mlruns/`) is intentionally ignored by git; the
reproducible report and comparison artifacts are committed instead.

## Containers and CI

Containers are optional; local Python commands are the primary experiment path. Validate
and start the training-dependent API service with:

```bash
docker compose -f docker/docker-compose.yml config --quiet
docker compose -f docker/docker-compose.yml up --build api
```

The `trainer` service completes first and writes the selected artifact into a shared volume;
the API starts only after training succeeds. Test it at <http://127.0.0.1:8000/docs> or with
the same `curl` requests shown above, then stop it with:

```bash
docker compose -f docker/docker-compose.yml down
```

The Compose configuration and local API behavior were verified; the optional vLLM container
was not run on Apple Silicon because it targets a Linux host with an NVIDIA GPU.

On pull requests, [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs lint and tests,
regenerates evaluation evidence, applies the macro-F1/fraud-recall promotion policy, uploads
the reports and plots, and comments the gate result on the PR. A passing candidate must
reproduce the champion or clear the configured improvement margin without regressing fraud
recall or inflating fold variance.

## Direct Qwen comparison

On Apple Silicon, the evaluated Qwen 2.5 1.5B Instruct model runs through Ollama's Metal
backend and OpenAI-compatible API:

```bash
ollama pull qwen2.5:1.5b-instruct
ollama serve
```

In another terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export LLM_MODEL=qwen2.5:1.5b-instruct
export LLM_BACKEND=ollama-metal
uv run support-router cv \
  --models llm_zero_shot,llm_few_shot \
  --schemes grouped \
  --append \
  --track
```

Each response is parsed and checked against the four allowed routes before it is scored.
Measured grouped-CV results were:

| approach | macro-F1 | `fraud-report` recall | invalid responses |
|---|---:|---:|---:|
| Qwen zero-shot | 0.762 | 0.840 | 0 / 1,600 |
| Qwen few-shot | 0.706 | 0.925 | 21 / 1,600 (1.3%) |

The LLM was strong on the three concrete problem routes but weak on negatively defined
`general` messages. It was retained as a comparison, not selected for default inference;
the frozen-embedding linear head was both more accurate and cheaper to serve.

The optional Docker vLLM service targets Linux with an NVIDIA GPU. It is not the local
Apple Silicon path. Both servers expose the same OpenAI-compatible interface, so the
classifier code does not change.

## Tests and validation

```bash
uv run ruff check .
uv run pytest -q
docker compose -f docker/docker-compose.yml config --quiet
```

The tests cover input and schema validation, grouping and leakage prevention, metric
calculation, model construction, LLM parsing/failure accounting, training and inference,
CSV scoring, API behavior, MLflow logging behavior, and the promotion gate.

## Repository layout

```text
src/support_router/   modeling, evaluation, tracking, CLI, and API
tests/                unit and integration tests
conf/                 reproducible experiment and promotion parameters
data/raw/train.csv    labeled training data
reports/              EDA, plots, comparisons, timing, and consolidated report
docker/               optional API/trainer and Linux/NVIDIA vLLM services
```

## Scaling and model choice

The committed benchmark records the hardware and measured warm-up, median, p95 and batch
throughput in [reports/latency.json](reports/latency.json). A target of 10,000
requests/minute is about 167/s, so a small warm CPU fleet behind a load balancer is the
default deployment. TF-IDF linear and tree models are cheaper still, but scored lower on
the honest split.

A generative LLM is justified when the route policy depends on broader context, rapidly
changing instructions, or reasoning that a fixed classifier cannot represent. It changes
the serving shape: GPU workers, continuous batching, bounded generation, prefix/KV
caching, strict output validation, deadlines, fallbacks, and cost-aware autoscaling. A
confidence cascade can keep the fast CPU classifier on the common path and send only
uncertain cases to the LLM or human review.

See [DEPLOYMENT.md](DEPLOYMENT.md) and the editable
[architecture.drawio](architecture.drawio) for the full 10,000 requests/minute design.

## Scope and trade-offs

Prioritized:

- honest leakage-resistant evaluation before model complexity;
- a strong classical baseline and several genuinely different comparisons;
- explicit fraud-recall protection alongside the primary metric;
- tested prediction, batch-scoring, and input-validation paths;
- reproducible experiment tracking and inspectable artifacts.

Deliberately left out:

- transformer fine-tuning on only 80 independent template groups;
- SMOTE or text resampling over mostly duplicated templates;
- an exhaustive hyperparameter search likely to overfit the CV folds;
- production authentication, tenancy, and automatic registry deployment;
- a calibrated human-review threshold, which needs a business cost model.

With more time, the priorities would be real-ticket validation, probability calibration,
targeted labeling around the fraud/dispute boundary, drift monitoring, and a shadow-mode
deployment before automated routing.

Active hands-on work took approximately **3–5 focused hours**, using AI assistance as
permitted by the brief; longer training and experiment runs completed unattended. Under a
strict three-hour budget, the stopping point would be grouped evaluation, TF-IDF linear
baselines, the prediction and batch-scoring interfaces, validation, tests, and a concise
decision note.

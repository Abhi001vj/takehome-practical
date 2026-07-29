# Support-ticket router

A four-class text classifier for crypto/fintech support messages. It routes each message
to `account-access`, `transaction-dispute`, `fraud-report`, or `general` and provides a
Python interface, CLI, batch scorer, and optional FastAPI service.

The central modeling decision is the evaluation split. The 400 messages are generated
from lightly varied templates: they collapse to 80 near-duplicate groups, and a normal
row-level split puts a template sibling in training for 95.8% of validation rows. Model
selection therefore uses repeated stratified group cross-validation. The naive split is
retained only to make the leakage gap visible.

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

## Prediction interfaces

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

The holdout-scoring entry point preserves input order and writes one prediction per valid
row:

```bash
uv run support-router score holdout.csv \
  --output predictions.csv \
  --confidence
```

Start the optional HTTP service with `uv run support-router serve`. Its OpenAPI UI is at
<http://127.0.0.1:8000/docs>, with `POST /predict`, `POST /predict/batch`, `GET /health`,
and `GET /info` endpoints.

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

The complete practice build took approximately **10–12 focused hours**. Under a strict
three-hour budget, the stopping point would be grouped evaluation, TF-IDF linear
baselines, the prediction and batch-scoring interfaces, validation, tests, and a concise
decision note.

# Reproduction commands

Run all commands from the repository root. The default comparison includes the local
scikit-learn and tree models. Frozen-embedding models are selected explicitly because they
require the optional LLM dependencies. Direct generative-model evaluation is separate and
requires a running OpenAI-compatible endpoint.

## Environment setup

```bash
uv sync --all-extras
```

## MLflow

Start the local tracking server in a separate terminal:

```bash
uv run mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --default-artifact-root "$PWD/mlruns" \
  --host 127.0.0.1 \
  --port 5001
```

Set the tracking URI in each terminal used to run experiments:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

Open <http://127.0.0.1:5001> and select the `support-routing` experiment to inspect
parameters, metrics, traces, datasets, and artifacts. Tracking is enabled by default. Add
`--no-track` only for temporary runs that should not be recorded.

## Complete experiment

```bash
uv run support-router test
uv run support-router experiment \
  --models all,embedding_logreg,embedding_lightgbm \
  --track \
  --register-winner
```

This generates the exploratory analysis, evaluates all selected models under grouped and
row-level cross-validation, writes the comparison artifacts, trains and benchmarks the best
grouped-CV model, runs the promotion gate, and generates `reports/REPORT.md`.

## Individual stages

```bash
uv run support-router eda
uv run support-router leakage
uv run support-router cv \
  --models all,embedding_logreg,embedding_lightgbm \
  --schemes grouped,naive \
  --track
uv run support-router train --model embedding_logreg --register --track
uv run support-router benchmark
uv run support-router gate --candidate-metrics reports/comparison.json
uv run support-router report
uv run support-router promote --version <VERSION>
```

Generated analysis is written under `reports/`. Trained model artifacts are written under
`artifacts/`.

## Classical model subsets

```bash
uv run support-router cv --models linear --schemes grouped
uv run support-router cv --models naive_bayes --schemes grouped
uv run support-router cv --models classical --schemes grouped
uv run support-router cv \
  --models logistic_regression,multinomial_nb,linear_svc \
  --schemes grouped
```

## Hyperparameter tuning

```bash
uv run support-router tune --models logistic_regression
uv run support-router tune --models logistic_regression,lightgbm
```

## Prediction interfaces

Train a model before running predictions:

```bash
uv run support-router train --model embedding_logreg
uv run support-router predict "Someone transferred ETH without my permission" --scores
uv run support-router score holdout.csv --output predictions.csv --confidence
uv run support-router serve
```

The HTTP service provides:

- `POST /predict`
- `POST /predict/batch`
- `GET /health`
- `GET /info`
- `GET /docs`

## Docker services

Docker is optional for the API and Linux/NVIDIA inference services. Experiments and MLflow
run locally by default. Build and run the API with:

```bash
docker compose -f docker/docker-compose.yml up --build api
```

Run the training container:

```bash
docker compose -f docker/docker-compose.yml up trainer
```

## Direct Qwen evaluation

On Apple Silicon, use the local Ollama Metal backend:

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

On a Linux host with an NVIDIA GPU, the optional vLLM service exposes the same
OpenAI-compatible interface:

```bash
docker compose -f docker/docker-compose.yml --profile llm up -d vllm
export LLM_BASE_URL=http://127.0.0.1:8001/v1
export LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
uv run support-router cv \
  --models llm_zero_shot,llm_few_shot \
  --schemes grouped \
  --append \
  --track
```

`LLM_MODEL` must match the identifier exposed by the selected server. The endpoint must
support OpenAI-compatible chat completions. Returned routes are validated against the four
supported labels before they are accepted.

Register the already-evaluated Qwen wrappers without repeating the complete network-bound
CV sweep:

```bash
uv run support-router train \
  --model llm_zero_shot \
  --no-evaluate \
  --metrics-from reports/comparison.json \
  --out artifacts/llm_zero_shot \
  --register \
  --track
uv run support-router train \
  --model llm_few_shot \
  --no-evaluate \
  --metrics-from reports/comparison.json \
  --out artifacts/llm_few_shot \
  --register \
  --track
```

The source comparison path is logged as an MLflow parameter and tag. This command packages
the serving wrapper and attaches its existing grouped-CV result; it does not claim a fresh
evaluation.

## Stop Docker services

```bash
docker compose -f docker/docker-compose.yml --profile llm down
```

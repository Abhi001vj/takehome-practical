# Support routing — end-to-end report

_Generated 2026-07-29 09:54 UTC by `support-router report`._

This report is assembled from the artifacts used by evaluation and promotion; it does not recompute or hand-edit model scores.

## Headline

| measure | result |
|---|---:|
| selected model | `embedding_logreg` |
| grouped macro-F1 | **0.971 ± 0.038** |
| `fraud-report` recall | **0.945** |
| rows / independent template groups | 400 / 80 |
| local MLflow runs / traces | 149 / 3600 |

The selection score is repeated grouped cross-validation. A random row split is reported only as a leakage diagnostic, never as the model-selection estimate.

## Data and leakage

400 rows -> 172 normalised forms -> 80 groups after similarity merge (largest group 14 rows, 80.0% of rows are template repeats)

| route | rows | template groups |
|---|---:|---:|
| `account-access` | 100 | 24 |
| `transaction-dispute` | 90 | 15 |
| `fraud-report` | 50 | 18 |
| `general` | 160 | 23 |

Row imbalance is **3.2:1**, but template-level imbalance is **1.6:1**. This is why class weighting is preferable to synthesizing more versions of duplicated templates.

| split | rows with a near-duplicate in training | leak rate |
|---|---:|---:|
| naive | 383 / 400 | 95.8% |
| grouped | 0 / 400 | 0.0% |

Full EDA: [EDA report](eda.md) · [class distribution](class_distribution.png) · [message lengths](length_distribution.png) · [template groups](template_group_sizes.png) · [word clouds](wordclouds.png) · [distinctive terms](distinctive_terms.png)

## Model comparison

| rank | model | grouped macro-F1 | std | fraud recall | accuracy | naive macro-F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `embedding_logreg` | **0.971** | 0.038 | 0.945 | 0.979 | 0.998 |
| 2 | `embedding_lightgbm` | **0.889** | 0.059 | 0.765 | 0.907 | 0.978 |
| 3 | `linear_svc` | **0.829** | 0.090 | 0.810 | 0.846 | 0.996 |
| 4 | `ridge` | **0.819** | 0.088 | 0.815 | 0.836 | 0.996 |
| 5 | `logistic_regression` | **0.808** | 0.088 | 0.780 | 0.830 | 0.997 |
| 6 | `passive_aggressive` | **0.790** | 0.103 | 0.790 | 0.813 | 0.997 |
| 7 | `bernoulli_nb` | **0.769** | 0.108 | 0.630 | 0.820 | 0.984 |
| 8 | `llm_zero_shot` | **0.762** | 0.086 | 0.840 | 0.728 | 0.769 |
| 9 | `random_forest` | **0.761** | 0.106 | 0.795 | 0.801 | 0.997 |
| 10 | `sgd` | **0.735** | 0.125 | 0.840 | 0.756 | 0.994 |
| 11 | `extra_trees` | **0.732** | 0.137 | 0.775 | 0.786 | 0.996 |
| 12 | `llm_few_shot` | **0.706** | 0.096 | 0.925 | 0.679 | 0.701 |
| 13 | `multinomial_nb` | **0.696** | 0.107 | 0.355 | 0.764 | 0.970 |
| 14 | `complement_nb` | **0.688** | 0.132 | 0.480 | 0.728 | 0.976 |
| 15 | `xgboost` | **0.679** | 0.081 | 0.545 | 0.719 | 0.950 |
| 16 | `catboost` | **0.669** | 0.090 | 0.570 | 0.709 | 0.966 |
| 17 | `lightgbm` | **0.669** | 0.083 | 0.545 | 0.708 | 0.962 |
| 18 | `knn` | **0.658** | 0.091 | 0.505 | 0.705 | 0.962 |
| 19 | `decision_tree` | **0.531** | 0.097 | 0.845 | 0.577 | 0.988 |
| 20 | `stratified_random` | **0.262** | 0.044 | 0.130 | 0.297 | 0.258 |
| 21 | `uniform_random` | **0.237** | 0.059 | 0.285 | 0.251 | 0.244 |
| 22 | `most_frequent` | **0.143** | 0.002 | 0.000 | 0.400 | 0.143 |

Detailed fold results and confusion matrices: [comparison.md](comparison.md) · [comparison.csv](comparison.csv) · [comparison.json](comparison.json)

### Direct Qwen experiment

Qwen 2.5 1.5B Instruct was evaluated through a local OpenAI-compatible Ollama endpoint using Apple Metal. It was evaluated by the same grouped folds; it was not used to select or generate labels for the training data.

| prompt | macro-F1 | std | fraud recall | parse failures | CV predict time |
|---|---:|---:|---:|---:|---:|
| `llm_zero_shot` | 0.762 | 0.086 | 0.840 | 0 (0.0%) | 1.953s |
| `llm_few_shot` | 0.706 | 0.096 | 0.925 | 21 (1.3%) | 8.863s |

The Qwen timing above is aggregate CV request time with deterministic response caching; it is not a clean online latency benchmark. The deployment decision is therefore based on accuracy, operating cost, and measured CPU winner latency.

## Inference performance

Measured on 60 unique messages after one warm-up:

| measure | value |
|---|---:|
| one-off load/warm-up | 3734.4 ms |
| median / p95 / max | 7.7 ms / 9.1 ms / 10.7 ms |
| single-message throughput | 127/s |
| batched throughput | 796/s |

At 10,000 requests/minute (about 167/s), the selected embedding-plus-linear model fits comfortably in a horizontally scaled CPU service. See [DEPLOYMENT.md](../DEPLOYMENT.md) and the editable [architecture.drawio](../architecture.drawio).

## MLflow evidence and registry

Tracking backend used for this report: `http://127.0.0.1:5001`. Experiment: `support-routing` (ID `2`).

| registered version | measured model | role |
|---:|---|---|
| 1 | `embedding_logreg` | comparison |
| 2 | `llm_zero_shot` | comparison |
| 3 | `llm_few_shot` | comparison |
| 4 | `embedding_logreg` | champion |
| 5 | `llm_zero_shot` | comparison |
| 6 | `llm_few_shot` | comparison |

The latest promotion-gate verdict is **PASS**. The configured candidate cleared the grouped-CV quality guardrails.

## Scope and trade-offs

Prioritized: leakage-resistant evaluation, macro-F1 plus an explicit fraud-recall guardrail, simple reproducible baselines, a callable prediction interface, batch holdout scoring, meaningful tests, and traceable experiment artifacts.

Deliberately left out: transformer fine-tuning on only 80 independent templates, SMOTE over duplicated text, a broad hyperparameter search, production auth and multi-tenancy, and automatic deployment from the model registry.

With more time: collect genuine tickets, calibrate confidence thresholds, add a human-review path for uncertain or high-risk decisions, monitor drift by route and confidence, and target labeling at the fraud/dispute boundary.

The complete practice build took approximately **10–12 focused hours**. A strict three-hour version would stop after grouped CV, TF-IDF linear baselines, the prediction/batch-scoring interfaces, validation, and tests.

## Required reasoning questions

### 1. Why macro-F1 when `fraud-report` is highest-stakes?

Accuracy is misleading here: predicting the majority `general` class gives 40% accuracy while finding no fraud. Macro-F1 gives each route equal weight and penalizes both missed tickets and bad routing. Because one average still cannot encode asymmetric harm, fraud recall is reported separately and enforced as a hard promotion floor. The operational extension is a calibrated threshold that sends ambiguous fraud-like cases to human review.

### 2. How was class imbalance handled, and how would harm be detected?

Linear classifiers use balanced class weights, folds preserve route balance while keeping template groups intact, and selection uses macro-F1. Resampling was avoided because most row imbalance comes from repeated templates. Harm would appear as low per-class recall/F1, especially fraud recall, a skewed confusion matrix, unstable fold results, or a changed prediction distribution in production.

### 3. Which decision was uncertain?

The uncertain decision was how aggressively to merge near-duplicate templates. Too low a similarity threshold can join genuinely different intents; too high a threshold leaks paraphrases across folds. Character 3–5-gram cosine similarity at 0.85 was a transparent compromise, followed by inspection of group size and the measured zero grouped leakage rate. With more data, the threshold would be sensitivity-tested.

### 4. What changes at 10,000 requests/minute or with an LLM?

At roughly 167 requests/s, the measured embedding-linear winner remains a CPU-first service: keep workers warm, batch embeddings briefly, cache normalized repeated text, and autoscale behind a load balancer. Pure TF-IDF linear/tree models are even cheaper but less accurate here. A generative LLM is appropriate when labels require broader context or change too quickly for retraining; it needs GPU serving, continuous batching, bounded tokens, prefix/KV caching, strict output validation, timeouts, and a fallback. A confidence-based cascade can reserve that cost for uncertain cases.

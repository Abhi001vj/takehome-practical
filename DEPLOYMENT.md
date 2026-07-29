# Deployment and scaling

The scale in the exercise is **10,000 requests per minute**, or about **167 requests per
second**. The same architecture
can grow toward the larger number, but the fleet size and failure isolation requirements
would be roughly sixty times greater.

## Recommended production design

Use a cascade rather than sending every ticket to a generative model:

1. Validate and normalise the request at the edge.
2. Check a versioned response cache for exact repeated messages.
3. Run the selected classifier on every cache miss.
4. Return high-confidence predictions immediately.
5. Send only low-confidence, out-of-distribution, long, or policy-sensitive messages to
   the generative model or Language Model absed models.
6. Route timeouts, invalid output, and ambiguous high-risk cases to human review.

This keeps the common path fast and deterministic while retaining the LLM for cases where
its flexibility adds value.

## Shared request path

```text
Clients
   |
CDN / WAF / rate limiting
   |
Regional load balancer
   |
API gateway: authentication, schema validation, request ID, 10k-character limit
   |
Versioned exact-response cache (hash of model version + normalised text)
   |
CPU classifier replicas
   |---- high confidence ------------------------------> route queue
   |
   `---- uncertain / long / novel ---> durable queue ---> GPU LLM pool
                                                   |       |
                                                   |       `-> constrained JSON result
                                                   `----------> human review on failure

Every path -> metrics, structured logs, traces, drift monitoring, delayed-label feedback
```

The service is stateless. Model artifacts are loaded before readiness succeeds, so replicas
can be added or removed without coordinating model state. The only shared online state is
the cache and queue; MLflow remains a control-plane system and is not on the prediction
request path.

## Deployment by model family

| Model family | Runtime | Hardware | Scaling unit | Main optimisation | When it is appropriate |
|---|---|---|---|---|---|
| TF-IDF + logistic/SVM/SGD/Ridge | sklearn process | CPU | stateless API replica | preloaded pipeline, multiple workers, exact-text cache | Best low-cost baseline; short fixed taxonomy and strict latency target |
| TF-IDF + tree/boosted tree | sklearn plus tree library | CPU | stateless API replica | cap threads per worker, control memory, batch sparse transforms | Useful only if measured gains justify larger artifacts and poorer sparse-text fit |
| Frozen embedding + linear head | encoder plus sklearn head | CPU by default; GPU optional | encoder worker or API replica | dynamic batching, embedding cache, quantisation where validated | Best measured quality here; semantic routing with deterministic class scores |
| Frozen embedding + tree head | encoder plus tree library | CPU/GPU encoder plus CPU head | encoder worker | share embeddings, batch encoder calls | Consider only when it beats the linear head; it did not in this experiment |
| Direct Qwen generation | vLLM-compatible inference server | production GPU pool | GPU replica | continuous batching, KV-cache sizing, prefix caching, constrained decoding | Fluid taxonomy, long multilingual input, entity extraction, rationale, or zero-labelled-data cold start |

### Linear and tree models on CPU

The vectoriser and estimator are serialized together and loaded once per process. Requests
do not need a feature store or database lookup. At 167 requests/second, a small fleet of
CPU replicas is enough; keep at least three across failure domains even if one instance can
handle the measured throughput.

Operational settings:

- use a process count based on physical cores rather than allowing every tree library to
  consume all threads inside every worker;
- bound the API queue so overload produces a controlled `429` or retry rather than
  unbounded latency;
- cache by `(model_version, normalised_text_hash)` so a deployment cannot return a stale
  label from an older model;
- autoscale on requests/second, queue depth, CPU, and p95 latency;
- keep the previous model version warm for immediate rollback.

Tree ensembles are not automatically more scalable because they are classical models.
They can be much larger than a linear head and make many branch-heavy memory accesses.
Their grouped-CV results must justify that serving cost; the current results do not.

### Frozen embeddings plus a small head

The encoder dominates latency; the logistic or tree head is negligible. The most useful
optimisations are therefore around encoding:

- micro-batch requests for 5-20 ms and run one encoder forward pass;
- cache the final route for exact repeats, or cache embeddings when several heads consume
  the same representation;
- separate the encoder into its own service only when multiple applications reuse it;
- use ONNX/Core ML/quantisation only after checking that grouped metrics and fraud recall
  remain within the promotion policy;
- warm the encoder before readiness and monitor cold-start time separately from steady
  state latency.

For this fixed four-route problem, the frozen encoder plus logistic head is the default
choice because it has the strongest grouped result, deterministic output, and useful class
scores for a review threshold.

## Generative Qwen serving

The repository talks to an OpenAI-compatible chat-completions endpoint. Development on
Apple Silicon can use Ollama/llama.cpp or the vLLM-Metal plugin. The Docker `vllm-openai`
image requests an NVIDIA device and is intended for a Linux GPU host, not Docker Desktop's
Metal GPU.

A production Qwen pool should include:

- an ingress queue that exposes queue time separately from model time;
- continuous batching so requests arriving together share GPU execution;
- a strict input length and a small output budget—the output here is one route;
- JSON/schema-constrained decoding and enum validation;
- prefix caching for the shared routing policy and, where supported, stable few-shot
  examples;
- an explicitly sized KV cache; longer context increases KV memory even when model weights
  are quantised;
- weight quantisation only after measuring quality on grouped folds, especially fraud
  recall;
- tensor parallelism for a model that does not fit one GPU and data-parallel replicas for
  throughput after it does;
- autoscaling on queued requests, queued tokens, time-to-first-token, tokens/second, p95
  latency, and GPU/KV-cache utilisation;
- timeout, retry budget, circuit breaker, and a deterministic fallback path;
- prompt/model version tags and sampled MLflow traces with sensitive text redacted or
  hashed in production.

Do not size the GPU fleet from requests/second alone. Estimate prompt and output tokens per
request, measure tokens/second per GPU at the real context-length distribution, and use:

```text
GPU replicas = ceil(peak tokens/second / measured tokens/second per GPU / target utilisation)
```

Keep target utilisation below saturation so queueing latency does not grow without bound.
Few-shot prompting has a materially larger shared prefix than zero-shot prompting; prefix
caching is therefore more valuable, but KV-cache and network costs are also higher.

## Choosing the small model or the LLM

The small classifier is the right choice when routes are stable, messages are short, labels
exist, throughput and cost matter, deterministic outputs are required, or class scores drive
a human-review threshold.

The generative model is the right choice when there is no labelled cold-start set, policy
changes faster than a retraining cycle, inputs are long or multilingual, the output needs
entities/severity/rationale as well as a label, or the uncertain tail is valuable enough to
justify GPU cost and additional failure modes.

For this dataset the recommended production decision is the cascade: the embedding-linear
model handles routine traffic and Qwen handles only uncertain or richer cases. A direct-LLM
result remains in the experiment table as measured evidence, not as the default deployment.

## Reliability and observability

Monitor both service health and model quality:

- request rate, error rate, p50/p95/p99 latency, queue depth, cache hit rate;
- predicted-route distribution and confidence distribution by model version;
- delayed per-class precision/recall when resolved labels arrive;
- `fraud-report` recall and fraud leak rate as release and production guardrails;
- invalid LLM output, timeouts, token counts, fallback rate, and human-review rate;
- data drift in message length, language, vocabulary, and embedding distance;
- canary results and rollback time during model deployment.

Deploy a candidate to a small traffic percentage, compare it with the champion, and promote
only after both technical SLOs and model guardrails pass. MLflow stores evaluation evidence
and model versions; the online service should resolve an approved immutable artifact rather
than querying experiment state on every request.

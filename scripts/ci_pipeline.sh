#!/usr/bin/env bash
# The single definition of "does this change deserve to merge".
#
# Run by the `ci` compose service and by the GitHub Actions workflow, so the two cannot
# drift. Any non-zero exit fails the PR.
#
# Stages, in dependency order:
#   1. lint + tests     - cheapest, fails fastest, and a broken harness invalidates
#                         every number produced downstream
#   2. EDA              - regenerates reports/eda.md so the report in the PR reflects
#                         the data actually committed, not a stale checkout
#   3. comparison       - grouped AND naive CV over the full registry
#   4. promotion gate   - the candidate must beat the registered champion
#
# The gate is stage 4 rather than stage 1 because a candidate that fails the gate is
# still worth retaining in the build artifact so the failure is diagnosable.

set -euo pipefail

MODELS="${SUPPORT_ROUTER_CI_MODELS:-all,embedding_logreg,embedding_lightgbm}"

echo "::group::lint"
ruff check src tests
echo "::endgroup::"

echo "::group::tests"
# `-m "not llm"` because CI has no vLLM endpoint. The embedding arms still run: MiniLM
# downloads once and runs on CPU, so only the *generative* arms are skipped.
pytest -q -m "not llm"
echo "::endgroup::"

echo "::group::eda"
support-router eda
echo "::endgroup::"

echo "::group::comparison"
support-router cv --models "$MODELS" --schemes grouped,naive
echo "::endgroup::"

echo "::group::promotion-gate"
# Exits non-zero when the candidate does not clear the champion by the configured
# margin, when fraud-report recall drops below the floor, or when fold variance has
# inflated. This is the check a branch-protection rule should require.
support-router gate --candidate-metrics reports/comparison.json
echo "::endgroup::"

echo "pipeline passed"

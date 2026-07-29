# Model comparison

## Dataset

- Rows: **400**
- Independent template groups: **80**
- 400 rows -> 172 normalised forms -> 80 groups after similarity merge (largest group 14 rows, 80.0% of rows are template repeats)

| label | rows | templates |
|---|---:|---:|
| account-access | 100 | 24 |
| transaction-dispute | 90 | 15 |
| fraud-report | 50 | 18 |
| general | 160 | 23 |

The row counts imply a 3.2:1 imbalance; the template counts imply roughly 1.6:1. Most of the apparent imbalance is duplication, not class rarity.

## Leakage

| scheme | test rows with a near-duplicate in train | rate |
|---|---:|---:|
| naive | 383 / 400 | 95.8% |
| grouped | 0 / 400 | 0.0% |

Under `naive` (plain StratifiedKFold) almost every test row has a template sibling in train, so its scores are inflated and do not transfer to the hidden holdout. `grouped` is zero by construction. **Select models on the grouped numbers.**

## Results (grouped CV)

| model | macro-F1 | ±std | fraud recall | accuracy | bal. acc | fit s | pred s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `embedding_logreg` | **0.971** | 0.038 | 0.945 | 0.979 | 0.974 | 0.61 | 0.004 |
| `embedding_lightgbm` | **0.889** | 0.059 | 0.765 | 0.907 | 0.887 | 1.64 | 0.001 |
| `linear_svc` | **0.829** | 0.090 | 0.810 | 0.846 | 0.845 | 0.03 | 0.004 |
| `ridge` | **0.819** | 0.088 | 0.815 | 0.836 | 0.838 | 0.02 | 0.004 |
| `logistic_regression` | **0.808** | 0.088 | 0.780 | 0.830 | 0.825 | 0.13 | 0.005 |
| `passive_aggressive` | **0.790** | 0.103 | 0.790 | 0.813 | 0.810 | 0.03 | 0.004 |
| `bernoulli_nb` | **0.769** | 0.108 | 0.630 | 0.820 | 0.772 | 0.00 | 0.001 |
| `llm_zero_shot` | **0.762** | 0.086 | 0.840 | 0.728 | 0.801 | 0.00 | 1.953 |
| `random_forest` | **0.761** | 0.106 | 0.795 | 0.801 | 0.774 | 0.39 | 0.051 |
| `sgd` | **0.735** | 0.125 | 0.840 | 0.756 | 0.768 | 0.03 | 0.004 |
| `extra_trees` | **0.732** | 0.137 | 0.775 | 0.786 | 0.754 | 0.24 | 0.042 |
| `llm_few_shot` | **0.706** | 0.096 | 0.925 | 0.679 | 0.787 | 0.00 | 8.863 |
| `multinomial_nb` | **0.696** | 0.107 | 0.355 | 0.764 | 0.701 | 0.00 | 0.001 |
| `complement_nb` | **0.688** | 0.132 | 0.480 | 0.728 | 0.706 | 0.00 | 0.001 |
| `xgboost` | **0.679** | 0.081 | 0.545 | 0.719 | 0.690 | 1.76 | 0.007 |
| `catboost` | **0.669** | 0.090 | 0.570 | 0.709 | 0.686 | 0.95 | 0.011 |
| `lightgbm` | **0.669** | 0.083 | 0.545 | 0.708 | 0.676 | 3.65 | 0.007 |
| `knn` | **0.658** | 0.091 | 0.505 | 0.705 | 0.667 | 0.02 | 0.020 |
| `decision_tree` | **0.531** | 0.097 | 0.845 | 0.577 | 0.595 | 0.01 | 0.001 |
| `stratified_random` | **0.262** | 0.044 | 0.130 | 0.297 | 0.263 | 0.00 | 0.000 |
| `uniform_random` | **0.237** | 0.059 | 0.285 | 0.251 | 0.249 | 0.00 | 0.000 |
| `most_frequent` | **0.143** | 0.002 | 0.000 | 0.400 | 0.250 | 0.00 | 0.000 |

## Results (naive CV)

| model | macro-F1 | ±std | fraud recall | accuracy | bal. acc | fit s | pred s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `embedding_logreg` | **0.998** | 0.009 | 0.990 | 0.999 | 0.997 | 0.00 | 0.000 |
| `logistic_regression` | **0.997** | 0.010 | 0.985 | 0.998 | 0.996 | 0.13 | 0.004 |
| `passive_aggressive` | **0.997** | 0.010 | 0.985 | 0.998 | 0.996 | 0.03 | 0.004 |
| `random_forest` | **0.997** | 0.010 | 0.985 | 0.998 | 0.996 | 0.29 | 0.041 |
| `linear_svc` | **0.996** | 0.012 | 0.980 | 0.997 | 0.995 | 0.03 | 0.004 |
| `ridge` | **0.996** | 0.012 | 0.980 | 0.997 | 0.995 | 0.02 | 0.004 |
| `extra_trees` | **0.996** | 0.010 | 0.980 | 0.997 | 0.995 | 0.23 | 0.042 |
| `sgd` | **0.994** | 0.013 | 0.970 | 0.996 | 0.993 | 0.03 | 0.004 |
| `decision_tree` | **0.988** | 0.026 | 0.980 | 0.991 | 0.988 | 0.01 | 0.001 |
| `bernoulli_nb` | **0.984** | 0.021 | 0.915 | 0.989 | 0.979 | 0.00 | 0.001 |
| `embedding_lightgbm` | **0.978** | 0.022 | 0.910 | 0.984 | 0.972 | 1.18 | 0.001 |
| `complement_nb` | **0.976** | 0.025 | 0.875 | 0.984 | 0.969 | 0.00 | 0.001 |
| `multinomial_nb` | **0.970** | 0.024 | 0.845 | 0.981 | 0.961 | 0.00 | 0.001 |
| `catboost` | **0.966** | 0.022 | 0.885 | 0.976 | 0.963 | 0.63 | 0.007 |
| `lightgbm` | **0.962** | 0.025 | 0.870 | 0.973 | 0.958 | 3.25 | 0.006 |
| `knn` | **0.962** | 0.026 | 0.830 | 0.972 | 0.951 | 0.01 | 0.017 |
| `xgboost` | **0.950** | 0.032 | 0.860 | 0.962 | 0.946 | 1.40 | 0.006 |
| `llm_zero_shot` | **0.769** | 0.028 | 0.840 | 0.728 | 0.801 | 0.00 | 0.001 |
| `llm_few_shot` | **0.701** | 0.047 | 0.910 | 0.676 | 0.781 | 0.00 | 8.248 |
| `stratified_random` | **0.258** | 0.054 | 0.145 | 0.292 | 0.259 | 0.00 | 0.000 |
| `uniform_random` | **0.244** | 0.052 | 0.265 | 0.259 | 0.253 | 0.00 | 0.000 |
| `most_frequent` | **0.143** | 0.000 | 0.000 | 0.400 | 0.250 | 0.00 | 0.000 |

## Best model (grouped CV)

`llm_zero_shot` - macro-F1 0.762 ± 0.086, fraud-report recall 0.840

Pooled confusion matrix (rows = truth, columns = prediction):

```
                       account-ac  transactio  fraud-repo     general
                                                                     
account-access                400           0           0           0
transaction-dispute             4         356           0           0
fraud-report                    8          24         168           0
general                       400           0           0         240
```

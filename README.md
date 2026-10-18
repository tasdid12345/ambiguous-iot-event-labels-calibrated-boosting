# Balanced, Calibrated LightGBM vs. Random Forest for Ambiguous IoT Event Interpretation

Process-mining systems that consume Internet-of-Things (IoT) data must often map a single sensor event to one of several possible activities. The paper "ProAmbitIon: Online Process Conformance Checking with Ambiguities Driven by the Internet of Things" (2023) motivates handling such ambiguity explicitly, rather than forcing each event to a single hard label. This experiment asks a narrower, measurable question about the classifier that produces those interpretations.

**Question:** Can a class-balanced, calibrated LightGBM classifier beat a standard random forest at resolving ambiguous smartphone sensor events into activity labels, measured by macro-F1 on the same held-out split?

**Dataset:** OpenML 1478, Human Activity Recognition with Smartphones (windowed inertial-sensor features, six activity classes), used as a public proxy for ambiguous IoT sensor events. This is not the paper's own dataset.

**Baseline:** a standard random forest. Its top-1 label is taken as the event interpretation.

**Improved model:** a class-balanced LightGBM with a small cross-validated hyperparameter search and sigmoid calibration. It also provides a top-2 candidate set for each event.

Both models use the same stratified 80/20 split, the same preprocessing, the same evaluation function and seeds 42–44. The main metric is macro-F1 on the test set. Top-1 accuracy and top-2 coverage are secondary metrics.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (and `figures/macro_f1_comparison.png`).

## Results (latest run)

On the OpenML 1478 smartphone HAR dataset, a standard random-forest top-1 event classifier reached 0.981 macro-F1, while a class-balanced, calibrated LightGBM reached 0.994 on the same held-out test split, a small gain of about 1.3 points.

| Metric | Value |
|---|---|
| baseline_score | 0.981 |
| baseline_score_std | 0.0008538 |
| improved_score | 0.9945 |
| improved_score_std | 0 |
| improvement | 0.01347 |
| baseline_f1 | 0.981 |
| improved_f1 | 0.9945 |
| baseline_top1_accuracy | 0.9811 |
| improved_top1_accuracy | 0.9942 |
| baseline_top2_coverage | 1 |
| improved_top2_coverage | 1 |
| improved_cv_macro_f1 | 0.9915 |

Run on 2026-10-18. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Proxy experiment: HAR sensor windows stand in for ambiguous IoT events; the paper's own data and model are not used. Ambiguity is not labelled in the data, so top-2 coverage is a diagnostic. Selected LightGBM params: {'num_leaves': 15, 'n_estimators': 400, 'min_child_samples': 30, 'learning_rate': 0.1}. Baseline top-2 coverage is derived from RF probabilities and is reported only for comparison. Std is the sample std over 3 seeds. The openml client controls its own HTTP timeouts; retries wrap the download calls.

## Reference

This project is motivated by the following paper:

> M Franceschetti, R Seiger, MJG González (2023). *ProAmbitIon: Online Process Conformance Checking with Ambiguities Driven by the Internet of Things.*. CAiSE Research.

Paper author: Prof. Barbara Weber, University of St.Gallen (barbara.weber@unisg.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

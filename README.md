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

## Reference

This project is motivated by the following paper:

> M Franceschetti, R Seiger, MJG González (2023). *ProAmbitIon: Online Process Conformance Checking with Ambiguities Driven by the Internet of Things.*. CAiSE Research.

Paper author: Prof. Barbara Weber, University of St.Gallen (barbara.weber@unisg.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

# History Before Algorithms: Governed AI for Accounting Control in Small Enterprises

Code and results for the paper:

> A. A. Y. Al-Saffar and W. M. Waadalla, "History Before Algorithms: Governed AI for Accounting Control in Small Enterprises," submitted to ICAIGS 2026, University of Mosul.

## What is here

| Path | Content |
| --- | --- |
| `src/late_payment.py` | Experiment 1: late-payment prediction (leakage-free, chronological 70/30 split) |
| `src/anomaly.py` | Experiment 2: invoice anomaly detection with injected labelled anomalies (10 seeds) |
| `results/late_payment.json` | All numbers reported for Experiment 1 |
| `results/anomaly.json` | All numbers reported for Experiment 2 |
| `data/` | Place the dataset here (not redistributed) |

## Live demo

An Arabic demo app of both AI control modules, with synthetic data: https://alialsaffarmosuluniversity.github.io/ai-receivables-control/ (source: `docs/index.html`).

## Data

The experiments use the public IBM **Finance Factoring – Late Payment Histories** dataset
(2,466 invoices, 100 customers). Download `WA_Fn-UseC_-Accounts-Receivable.csv` from
[Kaggle](https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories)
and save it as `data/ar.csv`.

## Reproduce

```bash
pip install -r requirements.txt
python src/late_payment.py data/ar.csv results/late_payment.json
python src/anomaly.py      data/ar.csv results/anomaly.json
```

Tested with Python 3, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3.

## Key results

| Experiment | Best method | Result |
| --- | --- | --- |
| Late-payment prediction | Logistic regression | ROC-AUC 0.882 (95% CI 0.854–0.907) |
| Invoice anomaly detection | Isolation Forest | F1 0.863, ROC-AUC 0.996 |

## License and citation

Released under the MIT License. You are free to use the code, provided the authors are credited.
Please cite the paper using `CITATION.cff`.

**Authors:** College of Administration and Economics, University of Mosul, Iraq

- Ali Abdulwahab Yahya Al-Saffar (corresponding), Department of Business Administration · ali_alsaffar@uomosul.edu.iq · ORCID [0009-0005-0918-893X](https://orcid.org/0009-0005-0918-893X)
- Waadalla Maan Waadalla, Department of Accounting · ORCID [0000-0003-4023-9254](https://orcid.org/0000-0003-4023-9254)

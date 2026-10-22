"""Late-payment prediction on the IBM Accounts Receivable dataset.
Leakage-free design: only information available at invoice issue time.
Customer-history features use only invoices SETTLED before the current invoice date.
Chronological 70/30 split.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.inspection import permutation_importance

SEED = 42
d = pd.read_csv(sys.argv[1])
for c in ["PaperlessDate", "InvoiceDate", "DueDate", "SettledDate"]:
    d[c] = pd.to_datetime(d[c], format="%m/%d/%Y")
d = d.sort_values(["InvoiceDate", "invoiceNumber"]).reset_index(drop=True)
d["late"] = (d.DaysLate > 0).astype(int)
d["disputed"] = (d.Disputed == "Yes").astype(int)

# invoice-time features
d["amount"] = d.InvoiceAmount
d["credit_term"] = (d.DueDate - d.InvoiceDate).dt.days
d["electronic"] = ((d.PaperlessBill == "Electronic") & (d.InvoiceDate >= d.PaperlessDate)).astype(int)
d["month"] = d.InvoiceDate.dt.month
d["weekday"] = d.InvoiceDate.dt.weekday
for cc in sorted(d.countryCode.unique()):
    d[f"country_{cc}"] = (d.countryCode == cc).astype(int)

# customer history from invoices settled strictly before this invoice date
hist = {k: [] for k in ["h_n", "h_late_rate", "h_mean_dayslate", "h_dispute_rate", "h_mean_amount"]}
for i, r in d.iterrows():
    p = d[(d.customerID == r.customerID) & (d.SettledDate < r.InvoiceDate)]
    n = len(p)
    hist["h_n"].append(n)
    hist["h_late_rate"].append(p.late.mean() if n else np.nan)
    hist["h_mean_dayslate"].append(p.DaysLate.mean() if n else np.nan)
    hist["h_dispute_rate"].append(p.disputed.mean() if n else np.nan)
    hist["h_mean_amount"].append(p.amount.mean() if n else np.nan)
for k, v in hist.items():
    d[k] = v
d["h_has_history"] = (d.h_n > 0).astype(int)

cut = int(len(d) * 0.7)
train, test = d.iloc[:cut].copy(), d.iloc[cut:].copy()
# fill missing history with training-set global values (no test info)
fills = {"h_late_rate": train.late.mean(), "h_mean_dayslate": train.DaysLate.mean(),
         "h_dispute_rate": train.disputed.mean(), "h_mean_amount": train.amount.mean()}
for df in (train, test):
    for k, v in fills.items():
        df[k] = df[k].fillna(v)
    df["amount_vs_hist"] = df.amount / df.h_mean_amount

country_cols = [c for c in d.columns if c.startswith("country_")]
FEATS = ["amount", "credit_term", "electronic", "month", "weekday", "h_n", "h_late_rate",
         "h_mean_dayslate", "h_dispute_rate", "h_has_history", "amount_vs_hist"] + country_cols

Xtr, ytr, Xte, yte = train[FEATS], train.late, test[FEATS], test.late

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED),
}

def metrics(y, p, s):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p), f1=f1_score(y, p), auc=roc_auc_score(y, s) if s is not None else 0.5,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def boot_auc(y, s, n=1000):
    rng = np.random.default_rng(SEED); y = np.asarray(y); s = np.asarray(s); out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        out.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

res = {"n_total": len(d), "n_customers": int(d.customerID.nunique()),
       "late_rate_all": float(d.late.mean()), "n_train": len(train), "n_test": len(test),
       "late_rate_train": float(ytr.mean()), "late_rate_test": float(yte.mean()),
       "train_period": [str(train.InvoiceDate.min().date()), str(train.InvoiceDate.max().date())],
       "test_period": [str(test.InvoiceDate.min().date()), str(test.InvoiceDate.max().date())],
       "amount_mean": float(d.amount.mean()), "amount_median": float(d.amount.median()),
       "dayslate_mean_given_late": float(d[d.late == 1].DaysLate.mean()),
       "dispute_rate": float(d.disputed.mean()),
       "late_rate_disputed": float(d[d.disputed == 1].late.mean()),
       "late_rate_undisputed": float(d[d.disputed == 0].late.mean()),
       "late_rate_electronic": float(d[d.electronic == 1].late.mean()),
       "late_rate_paper": float(d[d.electronic == 0].late.mean()),
       "features": FEATS, "models": {}}

# naive baselines
res["models"]["Majority class (all on time)"] = metrics(yte, np.zeros(len(yte), int), None)
s_prev = test.h_late_rate.values
res["models"]["Rule: previous late rate > 0.5"] = metrics(yte, (s_prev > 0.5).astype(int), s_prev)
res["models"]["Rule: previous late rate > 0.5"]["auc"] = roc_auc_score(yte, s_prev)

best = None
for name, m in models.items():
    m.fit(Xtr, ytr)
    s = m.predict_proba(Xte)[:, 1]
    p = (s >= 0.5).astype(int)
    r = metrics(yte, p, s)
    r["auc_ci95"] = boot_auc(yte, s)
    # business view: top 20% riskiest invoices
    k = int(0.2 * len(s)); top = np.argsort(-s)[:k]
    r["top20_precision"] = float(yte.values[top].mean())
    r["top20_capture"] = float(yte.values[top].sum() / yte.sum())
    res["models"][name] = r
    if best is None or r["auc"] > res["models"][best]["auc"]:
        best = name
res["best_model"] = best

pi = permutation_importance(models[best], Xte, yte, scoring="roc_auc", n_repeats=20, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=FEATS)
# collapse country dummies
imp_c = imp.drop(country_cols); imp_c["country (all codes)"] = imp[country_cols].sum()
res["perm_importance"] = imp_c.sort_values(ascending=False).round(4).to_dict()

# sensitivity: adding the Disputed flag (only known if dispute raised at issue)
Xtr2 = train[FEATS + ["disputed"]]; Xte2 = test[FEATS + ["disputed"]]
m2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED).fit(Xtr2, ytr)
s2 = m2.predict_proba(Xte2)[:, 1]
res["sensitivity_with_disputed"] = metrics(yte, (s2 >= .5).astype(int), s2)

json.dump(res, open(sys.argv[2], "w"), indent=2, default=float)
for k, v in res["models"].items():
    print(f"{k:35s} acc={v['accuracy']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} AUC={v['auc']:.3f}")
print("best", best, res["models"][best].get("auc_ci95"), res["models"][best].get("top20_capture"))
print(res["perm_importance"])
print("with disputed", {k: round(v, 3) for k, v in res["sensitivity_with_disputed"].items()})
print({k: res[k] for k in ["late_rate_all", "late_rate_train", "late_rate_test", "train_period", "test_period"]})

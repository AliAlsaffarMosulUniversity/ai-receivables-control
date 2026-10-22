"""Invoice anomaly detection with injected, labelled anomalies (10 random seeds).
Anomaly types (each ~1% of invoices, 3% total):
  A inflated amount   (x3 to x6 of the original)
  B unauthorised discount (amount x0.10 to x0.30)
  C duplicate invoice (copy of an existing invoice, same customer/amount, issued 0-2 days later)
Detectors: customer-level z-score rule, Isolation Forest, Local Outlier Factor, hybrid (rule OR IF).
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

base = pd.read_csv(sys.argv[1])
base["InvoiceDate"] = pd.to_datetime(base.InvoiceDate, format="%m/%d/%Y")
base = base[["customerID", "InvoiceDate", "InvoiceAmount"]].rename(columns={"InvoiceAmount": "amount"})
RATE = 0.01

def inject(seed):
    rng = np.random.default_rng(seed)
    d = base.copy(); d["label"] = 0; d["atype"] = "normal"
    n = len(d); k = int(round(RATE * n))
    idx = rng.choice(n, 2 * k, replace=False)
    a, b = idx[:k], idx[k:]
    d.loc[a, "amount"] *= rng.uniform(3, 6, k); d.loc[a, ["label", "atype"]] = [1, "A_inflated"]
    d.loc[b, "amount"] *= rng.uniform(0.10, 0.30, k); d.loc[b, ["label", "atype"]] = [1, "B_discount"]
    src = rng.choice(np.setdiff1d(np.arange(n), idx), k, replace=False)
    dup = base.loc[src].copy()
    dup["InvoiceDate"] = dup.InvoiceDate + pd.to_timedelta(rng.integers(0, 3, k), unit="D")
    dup["label"] = 1; dup["atype"] = "C_duplicate"
    return pd.concat([d, dup], ignore_index=True)

def features(d):
    d = d.copy()
    g = d.groupby("customerID").amount
    s, c, ss = g.transform("sum"), g.transform("count"), g.transform(lambda x: (x ** 2).sum())
    # leave-one-out customer mean / std
    m = (s - d.amount) / (c - 1)
    v = ((ss - d.amount ** 2) / (c - 1)) - m ** 2
    sd = np.sqrt(np.clip(v, 1e-9, None))
    d["z_cust"] = (d.amount - m) / sd
    d["ratio_cust"] = d.amount / m
    d["log_ratio"] = np.log(d.ratio_cust)
    # near-duplicates: same customer, same amount (to the cent), within 3 days
    d["amt_r"] = d.amount.round(2)
    d = d.sort_values(["customerID", "amt_r", "InvoiceDate"])
    gap = d.groupby(["customerID", "amt_r"]).InvoiceDate.diff().dt.days
    # flag only the later invoice of a near-identical pair (the original is legitimate)
    d["dup_flag"] = (gap <= 3).astype(int)
    return d.sort_index()

def score(y, flag, s):
    return dict(precision=precision_score(y, flag, zero_division=0), recall=recall_score(y, flag),
                f1=f1_score(y, flag), auc=roc_auc_score(y, s))

FEATS = ["log_ratio", "z_cust", "dup_flag"]
runs = []
for seed in range(10):
    d = features(inject(seed)); y = d.label.values
    X = StandardScaler().fit_transform(d[FEATS])
    cont = y.mean()
    out = {}
    # 1 rule: |z|>3 or ratio outside [0.33,3] or duplicate
    s_rule = np.maximum(np.abs(d.z_cust.values) / 3, d.dup_flag.values * 1.0)
    f_rule = ((np.abs(d.z_cust) > 3) | (d.ratio_cust > 3) | (d.ratio_cust < 1 / 3) | (d.dup_flag == 1)).astype(int).values
    out["Rule-based (z-score + duplicate check)"] = (f_rule, s_rule)
    # 2 Isolation Forest
    iso = IsolationForest(n_estimators=300, contamination=cont, random_state=seed).fit(X)
    out["Isolation Forest"] = ((iso.predict(X) == -1).astype(int), -iso.score_samples(X))
    # 3 LOF
    lof = LocalOutlierFactor(n_neighbors=35, contamination=cont); f = (lof.fit_predict(X) == -1).astype(int)
    out["Local Outlier Factor"] = (f, -lof.negative_outlier_factor_)
    # 4 amount-only Isolation Forest (no customer context) - ablation
    iso2 = IsolationForest(n_estimators=300, contamination=cont, random_state=seed).fit(d[["amount"]])
    out["Isolation Forest (amount only, no customer history)"] = ((iso2.predict(d[["amount"]]) == -1).astype(int), -iso2.score_samples(d[["amount"]]))
    r = {}
    for name, (f, s) in out.items():
        m = score(y, f, s)
        for t in ["A_inflated", "B_discount", "C_duplicate"]:
            mask = (d.atype == t).values
            m[f"recall_{t}"] = f[mask].mean()
        m["flagged"] = int(f.sum())
        r[name] = m
    r["_n"] = len(d); r["_anom"] = int(y.sum())
    runs.append(r)

names = [k for k in runs[0] if not k.startswith("_")]
summary = {"n_invoices_after_injection": runs[0]["_n"], "n_anomalies": runs[0]["_anom"], "seeds": 10, "methods": {}}
for nme in names:
    keys = runs[0][nme].keys()
    summary["methods"][nme] = {k: {"mean": float(np.mean([r[nme][k] for r in runs])), "sd": float(np.std([r[nme][k] for r in runs]))} for k in keys}
json.dump(summary, open(sys.argv[2], "w"), indent=2)
print(summary["n_invoices_after_injection"], summary["n_anomalies"])
for nme, m in summary["methods"].items():
    print(f"{nme:52s} " + " ".join(f"{k}={v['mean']:.3f}±{v['sd']:.3f}" for k, v in m.items() if k != 'flagged'), "flagged", round(m['flagged']['mean']))

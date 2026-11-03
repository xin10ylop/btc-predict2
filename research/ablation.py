"""Drop-one-family ablation on delay-adjusted targets. A family earns its place only if
removing it hurts out-of-sample log-loss in most folds."""
import sys, json, numpy as np, pandas as pd
from btcpred.validate import walk_forward
X = pd.read_parquet("data/X_dev.parquet"); T = pd.read_parquet("data/T_dev.parquet")
fam = pd.read_csv("data/families.csv", index_col=0).iloc[:, 0].to_dict()
h = int(sys.argv[1])
tgt = f"zd2_{h}"
fams = sorted(set(fam[c] for c in X.columns))
res_full, _ = walk_forward(X, T, h, sub=3, target=tgt, verbose=False)
rows = []
for f in fams:
    feats = [c for c in X.columns if fam[c] != f]
    r, _ = walk_forward(X, T, h, sub=3, target=tgt, feats=feats, verbose=False)
    d_ll = res_full.ll_skill_bp.values - r.ll_skill_bp.values   # >0: family helps
    d_auc = res_full.auc.values - r.auc.values
    rows.append(dict(family=f, n_feats=sum(fam[c] == f for c in X.columns),
                     d_ll_mean=d_ll.mean(), folds_helped=int((d_ll > 0).sum()),
                     d_auc_mean=d_auc.mean() * 100, d_ll_per_fold=np.round(d_ll, 1).tolist()))
    print(rows[-1], flush=True)
print(f"\nFULL h={h}: ll_skill={res_full.ll_skill_bp.mean():.2f} auc={res_full.auc.mean():.4f} per-fold ll={np.round(res_full.ll_skill_bp.values,1).tolist()}")
D = pd.DataFrame(rows).sort_values("d_ll_mean", ascending=False)
print(D.drop(columns="d_ll_per_fold").round(3).to_string())
D.to_csv(f"reports/ablation_h{h}.csv", index=False)

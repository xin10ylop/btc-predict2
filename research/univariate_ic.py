"""Univariate rank-IC of every feature vs next-1m and next-5m vol-normalized return, per year.
A feature is interesting only if its IC has the SAME sign in (almost) every year."""
import numpy as np, pandas as pd
X = pd.read_parquet("data/X_dev.parquet"); T = pd.read_parquet("data/T_dev.parquet")
yr = X.index.year
rows = []
for h in (1, 5):
    z = T[f"z{h}"].clip(-6, 6)
    for y in sorted(set(yr)):
        msk = (yr == y) & z.notna().values
        zr = z[msk].rank(pct=True)
        Xr = X[msk].rank(pct=True)
        ic = Xr.corrwith(zr)
        for f, v in ic.items():
            rows.append((h, y, f, v))
D = pd.DataFrame(rows, columns=["h", "y", "f", "ic"])
for h in (1, 5):
    t = D[D.h == h].pivot(index="f", columns="y", values="ic") * 100
    t["mean"] = t.mean(1); t["consist"] = np.sign(t.iloc[:, :6]).mean(1) * np.sign(t["mean"])
    t = t.reindex(t["mean"].abs().sort_values(ascending=False).index)
    print(f"\n==== horizon {h}m: rank IC x100 per year ====")
    print(t.round(2).head(30).to_string())

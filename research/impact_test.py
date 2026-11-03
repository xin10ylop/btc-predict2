"""Joint regression per year: z(t->t+5, act +2s) ~ flow-explained part + flow-free residual of the
last 15 minutes' return (both in sigma units). Hypothesis (propagator / transient impact):
the flow-explained part reverts (beta<0), the flow-free part does not."""
import numpy as np, pandas as pd
X = pd.read_parquet("data/X_dev.parquet", columns=["impact_expl_15", "impact_resid_15", "impact_expl_60", "impact_resid_60"])
T = pd.read_parquet("data/T_dev.parquet", columns=["zd2_5", "zd2_1"])
D = X.join(T).dropna()
D = D[(D.abs() < 8).all(axis=1)]
for w in (15, 60):
    print(f"\nwindow {w}m   beta_expl  t_expl   beta_resid  t_resid   (target 5m, act+2s)")
    for y, g in D.groupby(D.index.year):
        A = np.column_stack([np.ones(len(g)), g[f"impact_expl_{w}"], g[f"impact_resid_{w}"]])
        yv = g["zd2_5"].values
        b, *_ = np.linalg.lstsq(A, yv, rcond=None)
        e = yv - A @ b
        # Newey-West-ish: inflate SE by sqrt(5) for overlapping 5m targets
        se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * e.var()) * np.sqrt(5)
        print(f"{y}   {b[1]:+.4f}  {b[1]/se[1]:+6.1f}    {b[2]:+.4f}  {b[2]/se[2]:+6.1f}")

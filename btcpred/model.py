"""The production predictor.

For each horizon h in 1..5 minutes it holds
  * a direction model  : P(price at close(t+h) > price you can get now), LightGBM binary,
                         averaged over several training windows (temporal ensemble)
  * a scale model      : expected |move| in units of the running EWMA sigma (LightGBM on log|z|)
  * the empirical distribution of standardised residuals -> quantile bands.

"now" = DELAY seconds after the close of the last 1-minute candle (default 2s), because
that is when a real bot can act; see reports for how the edge decays with delay.
"""
import json, os, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

HORIZONS = (1, 2, 3, 4, 5)
DIR_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=3000,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
                  max_bin=127, verbose=-1, num_threads=4)
VOL_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=3000,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
                  max_bin=127, verbose=-1, num_threads=4)
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
QGRID = np.linspace(0.01, 0.99, 99)
VOL_C = 0.05


def _fit(params, X, y, rounds, seed=0, return_es=False):
    """Early-stop on the most recent 10% of rows, then refit on all rows with that many trees."""
    n = len(y)
    cut = int(n * 0.9)
    p = {**params, "seed": seed}
    d1 = lgb.Dataset(X[:cut], y[:cut])
    d2 = lgb.Dataset(X[cut:], y[cut:], reference=d1)
    m = lgb.train(p, d1, rounds, valid_sets=[d2], callbacks=[lgb.early_stopping(60, verbose=False)])
    best = max(m.best_iteration, 20)
    full = lgb.train(p, lgb.Dataset(X, y), best)
    return (full, m) if return_es else full


class BTCPredictor:
    def __init__(self, features, delay=2, windows=(12, 24, 36), sub=3):
        self.features = list(features)
        self.delay = delay
        self.windows = tuple(windows)
        self.sub = sub
        self.dir_models = {}   # h -> list of boosters
        self.vol_models = {}   # h -> booster
        self.resid_q = {}      # h -> quantiles of standardized residual z / scale
        self.conf_q = {}       # h -> percentiles of |P(up) - 0.5| on recent training data
        self.meta = {}

    # ------------------------------------------------------------------ train
    def fit(self, X, T, end, verbose=True):
        """Train on data strictly before `end` (Timestamp/str)."""
        end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        idx = X.index
        for h in HORIZONS:
            z = T[f"zd{self.delay}_{h}"].values
            ok = np.isfinite(z)
            stop = end - pd.Timedelta(minutes=h + 1)
            self.dir_models[h] = []
            for w in self.windows:
                tr = np.flatnonzero(ok & (z != 0) & (idx >= end - pd.DateOffset(months=w)) & (idx < stop))[::self.sub]
                m = _fit(DIR_PARAMS, X[self.features].values[tr], (z[tr] > 0).astype(int), 400, seed=w)
                self.dir_models[h].append(m)
                if verbose:
                    print(f"h={h} window={w}m rows={len(tr)} trees={m.num_trees()}", flush=True)
            # historical distribution of |p - 0.5| -> lets live output say "top x% confidence"
            recent = np.flatnonzero(ok & (idx >= stop - pd.DateOffset(months=2)) & (idx < stop))[::5]
            pr = np.mean([m.predict(X[self.features].values[recent]) for m in self.dir_models[h]], axis=0)
            self.conf_q[h] = np.quantile(np.abs(pr - 0.5), np.linspace(0, 1, 101)).tolist()
            # scale model on the middle window
            w = sorted(self.windows)[len(self.windows) // 2]
            tr = np.flatnonzero(ok & (idx >= end - pd.DateOffset(months=w)) & (idx < stop))[::self.sub]
            ya = np.log(np.abs(z[tr]) + VOL_C)
            vm, vm_es = _fit(VOL_PARAMS, X[self.features].values[tr], ya, 600, return_es=True)
            self.vol_models[h] = vm
            # residual shape measured out-of-sample: early-stopping model on the held-out last 10%
            tail = tr[int(len(tr) * 0.9):]
            scale = self._scale(vm_es, X[self.features].values[tail])
            e = z[tail] / scale
            self.resid_q[h] = (np.quantile(e, QGRID) - np.median(e)).tolist()
        self.meta = dict(trained_until=str(end), delay_s=self.delay, windows=self.windows,
                         n_features=len(self.features))
        return self

    @staticmethod
    def _scale(vm, Xv):
        # any monotone map works because residual quantiles are measured in the same units
        return np.clip(np.exp(vm.predict(Xv)), 1e-3, None)

    # ---------------------------------------------------------------- predict
    def predict_frame(self, X):
        """Vectorised predictions for every row of X. Returns a DataFrame."""
        Xv = X[self.features].values
        out = {}
        for h in HORIZONS:
            ps = np.mean([m.predict(Xv) for m in self.dir_models[h]], axis=0)
            out[f"p_up_{h}"] = ps
            out[f"scale_{h}"] = self._scale(self.vol_models[h], Xv)
        return pd.DataFrame(out, index=X.index)

    def describe(self, X_row, price_now, sigma_now):
        """Human-readable forecast for the latest row.
        sigma_now: current 1-minute EWMA volatility (log units) from make_features."""
        f = self.predict_frame(X_row.iloc[[-1]]).iloc[0]
        res = {}
        for h in HORIZONS:
            unit = f[f"scale_{h}"] * sigma_now * np.sqrt(h)    # log-return per residual unit
            Q = np.array(self.resid_q[h])                       # centred residual quantiles
            p = float(f[f"p_up_{h}"])
            # location shift delta such that P(e + delta > 0) = p  (fixed shape, shifted)
            delta = -np.interp(1 - p, QGRID, Q)
            band = {f"q{int(round(qq*100))}": float((np.interp(qq, QGRID, Q) + delta) * unit * 1e4) for qq in QS}
            cq = self.conf_q.get(h)
            pct = float(np.interp(abs(p - 0.5), cq, np.linspace(0, 100, 101))) if cq else float("nan")
            res[h] = dict(p_up=p, confidence_pct=pct, median_move_bp=float(delta * unit * 1e4), band_bp=band,
                          price_median=float(price_now * np.exp(delta * unit)),
                          price_q05=float(price_now * np.exp(band["q5"] / 1e4)),
                          price_q95=float(price_now * np.exp(band["q95"] / 1e4)))
        return res

    # ------------------------------------------------------------------- io
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        for h, ms in self.dir_models.items():
            for i, m in enumerate(ms):
                m.save_model(os.path.join(folder, f"dir_h{h}_{i}.txt"))
        for h, m in self.vol_models.items():
            m.save_model(os.path.join(folder, f"vol_h{h}.txt"))
        json.dump(dict(features=self.features, delay=self.delay, windows=list(self.windows),
                       resid_q={str(k): v for k, v in self.resid_q.items()},
                       conf_q={str(k): v for k, v in self.conf_q.items()}, meta=self.meta),
                  open(os.path.join(folder, "predictor.json"), "w"), indent=1)

    @classmethod
    def load(cls, folder):
        cfg = json.load(open(os.path.join(folder, "predictor.json")))
        obj = cls(cfg["features"], cfg["delay"], tuple(cfg["windows"]))
        obj.resid_q = {int(k): v for k, v in cfg["resid_q"].items()}
        obj.conf_q = {int(k): v for k, v in cfg.get("conf_q", {}).items()}
        obj.meta = cfg["meta"]
        for h in HORIZONS:
            obj.dir_models[h] = [lgb.Booster(model_file=os.path.join(folder, f"dir_h{h}_{i}.txt"))
                                 for i in range(len(obj.windows))]
            obj.vol_models[h] = lgb.Booster(model_file=os.path.join(folder, f"vol_h{h}.txt"))
        return obj

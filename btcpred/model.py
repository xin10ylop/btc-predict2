"""The production predictor.

Horizons: "s10", "s15" (seconds after acting) and 1..5 (minutes, to the candle close h minutes
after the last closed candle). For each horizon it holds
  * a direction model  : P(price at close(t+h) > price you can get now), LightGBM binary,
                         averaged over several training windows (temporal ensemble)
  * a scale model      : expected |move| in units of the running EWMA sigma (LightGBM on log|z|)
  * the empirical distribution of standardised residuals -> quantile bands.

"now" = DELAY seconds after the close of the last 1-minute candle (default 2s), because
that is when a real bot can act; see reports for how the edge decays with delay.
"""
import gzip, json, os
import numpy as np
import pandas as pd
import lightgbm as lgb

MIN_HORIZONS = (1, 2, 3, 4, 5)       # minutes: target = close of candle t+h
SEC_HORIZONS = ("s10", "s15")        # seconds: target = price H seconds after acting (close + delay + H)
HORIZONS = SEC_HORIZONS + MIN_HORIZONS


def is_sec(h):
    return isinstance(h, str) and h.startswith("s")


def hkey(k):
    """JSON keys back to horizon keys: "3" -> 3, "s10" -> "s10"."""
    return int(k) if str(k).isdigit() else str(k)


def h_minutes(h):
    return int(h[1:]) / 60 if is_sec(h) else h


def h_label(h):
    return f"{h[1:]}s" if is_sec(h) else f"{h}m"


def target_col(h, delay):
    return f"zs{delay}_{h[1:]}" if is_sec(h) else f"zd{delay}_{h}"
DIR_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=3000,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
                  max_bin=127, verbose=-1, num_threads=4)
VOL_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=3000,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0,
                  max_bin=127, verbose=-1, num_threads=4)
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
QGRID = np.linspace(0.01, 0.99, 99)
VOL_C = 0.05


def _fit(params, X, y, rounds, seed=0, return_es=False, weight=None):
    """Early-stop on the most recent 10% of rows, then refit on all rows with that many trees."""
    n = len(y)
    cut = int(n * 0.9)
    p = {**params, "seed": seed}
    d1 = lgb.Dataset(X[:cut], y[:cut], weight=None if weight is None else weight[:cut])
    d2 = lgb.Dataset(X[cut:], y[cut:], reference=d1)
    m = lgb.train(p, d1, rounds, valid_sets=[d2], callbacks=[lgb.early_stopping(60, verbose=False)])
    best = max(m.best_iteration, 20)
    full = lgb.train(p, lgb.Dataset(X, y, weight=weight), best)
    return (full, m) if return_es else full


class BTCPredictor:
    def __init__(self, features, delay=2, windows=(12, 24, 36), sub=3, hfeatures=None, halflife=None):
        # hfeatures: optional {h: [features]} -- e.g. regime-state features only where they help
        self.hfeatures = {hkey(h): list(f) for h, f in (hfeatures or {}).items()}
        self.base = list(features)  # used by horizons without their own list
        self.features = list(dict.fromkeys(self.base + [f for fl in self.hfeatures.values() for f in fl]))
        self.halflife = halflife  # months; recency weighting of training rows (None = equal weights)
        self.delay = delay
        self.windows = tuple(windows)
        self.sub = sub
        self.dir_models = {}   # h -> list of boosters
        self.vol_models = {}   # h -> booster
        self.resid_q = {}      # h -> quantiles of standardized residual z / scale
        self.conf_q = {}       # h -> percentiles of |P(up) - 0.5| on recent training data
        self.calib = {}        # h -> {"temp": a, "band_k": k}: logit(p) *= a ; scale *= k
        self.track = {}        # h -> recent live-proxy track record (set by online_calibrate)
        self.flat_tab = {}     # seconds horizons: P(price unchanged) by decile of predicted scale
        self.meta = {}
        self.horizons = HORIZONS  # horizons this predictor holds (older saved models: minutes only)

    # ------------------------------------------------------------------ train
    def fit(self, X, T, end, verbose=True, horizons=HORIZONS):
        """Train on data strictly before `end` (Timestamp/str)."""
        end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        idx = X.index
        self.horizons = tuple(horizons)
        for h in self.horizons:
            fh = self.feats(h)
            z = T[target_col(h, self.delay)].values
            ok = np.isfinite(z)
            stop = end - pd.Timedelta(minutes=int(np.ceil(h_minutes(h))) + 1)
            self.dir_models[h] = []
            for w in self.windows:
                tr = np.flatnonzero(ok & (z != 0) & (idx >= end - pd.DateOffset(months=w)) & (idx < stop))[::self.sub]
                wt = None
                if self.halflife:
                    wt = 0.5 ** (((stop - idx[tr]).days.values / 30.44) / self.halflife)
                m = _fit(DIR_PARAMS, X[fh].values[tr], (z[tr] > 0).astype(int), 400, seed=w, weight=wt)
                self.dir_models[h].append(m)
                if verbose:
                    print(f"h={h} window={w}m rows={len(tr)} trees={m.num_trees()}", flush=True)
            # historical distribution of |p - 0.5| -> lets live output say "top x% confidence"
            recent = np.flatnonzero(ok & (idx >= stop - pd.DateOffset(months=2)) & (idx < stop))[::5]
            pr = np.mean([m.predict(X[fh].values[recent]) for m in self.dir_models[h]], axis=0)
            self.conf_q[h] = np.quantile(np.abs(pr - 0.5), np.linspace(0, 1, 101)).tolist()
            # scale model on the middle window
            w = sorted(self.windows)[len(self.windows) // 2]
            tr = np.flatnonzero(ok & (idx >= end - pd.DateOffset(months=w)) & (idx < stop))[::self.sub]
            ya = np.log(np.abs(z[tr]) + VOL_C)
            vm, vm_es = _fit(VOL_PARAMS, X[fh].values[tr], ya, 600, return_es=True)
            self.vol_models[h] = vm
            # residual shape measured out-of-sample: early-stopping model on the held-out last 10%
            tail = tr[int(len(tr) * 0.9):]
            scale = self._scale(vm_es, X[fh].values[tail])
            e = z[tail] / scale
            self.resid_q[h] = (np.quantile(e, QGRID) - np.median(e)).tolist()
            if is_sec(h):  # 10-18% of 10s windows have no price change: measure P(flat) vs predicted scale
                edges = np.quantile(scale, np.linspace(0, 1, 11))
                b = np.clip(np.searchsorted(edges, scale, side="right") - 1, 0, 9)
                self.flat_tab[h] = dict(edges=edges.tolist(), flat=[float(np.mean(z[tail][b == i] == 0)) for i in range(10)])
        self.meta = dict(trained_until=str(end), delay_s=self.delay, windows=self.windows,
                         n_features={str(h): len(self.feats(h)) for h in self.horizons}, halflife=self.halflife)
        return self

    def feats(self, h):
        return self.hfeatures.get(h, self.base)

    @staticmethod
    def _scale(vm, Xv):
        # any monotone map works because residual quantiles are measured in the same units
        return np.clip(np.exp(vm.predict(Xv)), 1e-3, None)

    # ---------------------------------------------------------------- predict
    def predict_frame(self, X, raw=False):
        """Vectorised predictions for every row of X. Returns a DataFrame.
        raw=True skips the probability temperature (used for online recalibration)."""
        out = {}
        for h in self.horizons:
            Xv = X[self.feats(h)].values
            ps = np.mean([m.predict(Xv) for m in self.dir_models[h]], axis=0)
            c = self.calib.get(h, {})
            a = 1.0 if raw else c.get("temp", 1.0)
            if a != 1.0:  # averaging probabilities shrinks them toward 0.5; undo with a temperature
                lg = np.log(ps / (1 - ps))
                ps = 1 / (1 + np.exp(-a * lg))
            out[f"p_up_{h}"] = ps
            out[f"scale_{h}"] = self._scale(self.vol_models[h], Xv) * c.get("band_k", 1.0)
        return pd.DataFrame(out, index=X.index)

    def describe(self, X_row, price_now, sigma_now):
        """Human-readable forecast for the latest row.
        sigma_now: current 1-minute EWMA volatility (log units) from make_features."""
        f = self.predict_frame(X_row.iloc[[-1]]).iloc[0]
        res = {}
        for h in self.horizons:
            unit = f[f"scale_{h}"] * sigma_now * np.sqrt(h_minutes(h))   # log-return per residual unit
            Q = np.array(self.resid_q[h])                       # centred residual quantiles
            p = float(f[f"p_up_{h}"])
            # location shift delta such that P(e + delta > 0) = p  (fixed shape, shifted)
            delta = -np.interp(1 - p, QGRID, Q)
            band = {f"q{int(round(qq*100))}": float((np.interp(qq, QGRID, Q) + delta) * unit * 1e4) for qq in QS}
            cq = self.conf_q.get(h)
            pct = float(np.interp(abs(p - 0.5), cq, np.linspace(0, 100, 101))) if cq else float("nan")
            p_flat = float("nan")
            if h in self.flat_tab:
                raw_scale = f[f"scale_{h}"] / self.calib.get(h, {}).get("band_k", 1.0)
                t = self.flat_tab[h]
                i = int(np.clip(np.searchsorted(t["edges"], raw_scale, side="right") - 1, 0, 9))
                p_flat = t["flat"][i]
            res[h] = dict(p_up=p, p_flat=p_flat, confidence_pct=pct, median_move_bp=float(delta * unit * 1e4), band_bp=band,
                          price_median=float(price_now * np.exp(delta * unit)),
                          price_q05=float(price_now * np.exp(band["q5"] / 1e4)),
                          price_q95=float(price_now * np.exp(band["q95"] / 1e4)))
        return res

    # ------------------------------------------------------- regime adaptation
    # live proxy (close-to-close) gives slightly sharper temperatures than the act+2s targets;
    # factors measured on the 2023-2026 walk-forward (research/online_recal_proxy.py)
    PROXY_ADJ = {1: 0.87, 2: 0.89, 3: 0.91, 4: 0.93, 5: 0.95}
    ONLINE_H = (1,)   # horizons that always use the 7-day online temperature (see README: regimes)

    def online_calibrate(self, X, close, days=7, min_rows=2000):
        """Refit each horizon's probability temperature on the last `days` days of realised
        outcomes (close(t+h) vs close(t)), using only rows whose outcome is already known.
        A broken edge drives the temperature toward 0 (calls shrink to 50%); a strong one
        sharpens them. Also records a recent track record per horizon."""
        from scipy.optimize import minimize_scalar
        start = X.index[-1] - pd.Timedelta(days=days)
        cutoff = pd.Timestamp(self.meta["trained_until"]) if self.meta.get("trained_until") else None
        if cutoff is not None and cutoff > start:  # never self-check on minutes the model trained on
            start = cutoff
        X = X[X.index >= start]
        if len(X) < min_rows:
            return self.track
        days = round((X.index[-1] - X.index[0]).total_seconds() / 86400, 1)
        F = self.predict_frame(X, raw=True)
        c = close.reindex(X.index)
        for h in [h for h in self.horizons if not is_sec(h)]:  # seconds horizons keep static calibration
            fut = close.shift(-h).reindex(X.index)
            ok = (fut.notna() & c.notna() & (fut != c) & F[f"p_up_{h}"].notna()).values
            fh = self.feats(h)
            ok = ok & (X[fh].notna().sum(axis=1).values >= 0.9 * len(fh))  # feature warm-up complete
            if ok.sum() < min_rows:
                continue
            p = np.clip(F[f"p_up_{h}"].values[ok], 1e-6, 1 - 1e-6)
            y = (fut.values[ok] > c.values[ok]).astype(float)
            lg = np.log(p / (1 - p))
            def nll(a):
                q = np.clip(1 / (1 + np.exp(-a * lg)), 1e-6, 1 - 1e-6)
                return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
            a = float(minimize_scalar(nll, bounds=(0.0, 3.0), method="bounded").x) * self.PROXY_ADJ[h]
            br = y.mean()
            ll0 = -(br * np.log(br) + (1 - br) * np.log(1 - br))
            static = self.calib.get(h, {}).get("static_temp", self.calib.get(h, {}).get("temp", 1.0))
            raw_skill = float((ll0 - nll(1.0)) * 1e4)
            # h in ONLINE_H: always follow the recent regime (helped in every test).
            # other horizons: circuit breaker, only when the recent raw edge is negative.
            use = a if (h in self.ONLINE_H or raw_skill < 0) else static
            self.calib.setdefault(h, {})["static_temp"] = static
            self.calib[h]["temp"] = round(use, 4)
            self.track[h] = dict(n=int(ok.sum()), hit=float(np.mean((p > .5) == (y > 0))),
                                 skill_bp=raw_skill, temp=use, online=bool(use == a), days=days)
        return self.track

    # ------------------------------------------------------------------- io
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        def dump(m, name):
            with gzip.open(os.path.join(folder, name + ".txt.gz"), "wt") as f:
                f.write(m.model_to_string())
        for h, ms in self.dir_models.items():
            for i, m in enumerate(ms):
                dump(m, f"dir_h{h}_{i}")
        for h, m in self.vol_models.items():
            dump(m, f"vol_h{h}")
        json.dump(dict(features=self.base, delay=self.delay, windows=list(self.windows),
                       hfeatures={str(k): v for k, v in self.hfeatures.items()}, halflife=self.halflife,
                       resid_q={str(k): v for k, v in self.resid_q.items()},
                       conf_q={str(k): v for k, v in self.conf_q.items()},
                       calib={str(k): v for k, v in self.calib.items()},
                       flat_tab={str(k): v for k, v in self.flat_tab.items()}, meta=self.meta),
                  open(os.path.join(folder, "predictor.json"), "w"), indent=1)

    @classmethod
    def load(cls, folder):
        cfg = json.load(open(os.path.join(folder, "predictor.json")))
        obj = cls(cfg["features"], cfg["delay"], tuple(cfg["windows"]), hfeatures=cfg.get("hfeatures"),
                  halflife=cfg.get("halflife"))
        obj.resid_q = {hkey(k): v for k, v in cfg["resid_q"].items()}
        obj.conf_q = {hkey(k): v for k, v in cfg.get("conf_q", {}).items()}
        obj.calib = {hkey(k): v for k, v in cfg.get("calib", {}).items()}
        obj.flat_tab = {hkey(k): v for k, v in cfg.get("flat_tab", {}).items()}
        obj.horizons = tuple(h for h in HORIZONS if h in obj.resid_q)
        obj.meta = cfg["meta"]
        def read(name):
            with gzip.open(os.path.join(folder, name + ".txt.gz"), "rt") as f:
                return lgb.Booster(model_str=f.read())
        for h in obj.horizons:
            obj.dir_models[h] = [read(f"dir_h{h}_{i}") for i in range(len(obj.windows))]
            obj.vol_models[h] = read(f"vol_h{h}")
        return obj

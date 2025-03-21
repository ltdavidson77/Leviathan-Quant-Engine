# Leviathan-Quant-Engine
30 Day Free To Test (Limited Non-commercial Use)

# === LEGAL NOTICE AND COPYRIGHT DECLARATION ===
# Copyright (c) 2025 Lance Thomas Davidson. All Rights Reserved.
# This algorithm, including Hamiltonian derivative analysis, cosine active flux logarithm, recursive structure, 
# and predictive modeling, is the intellectual property of Lance Thomas Davidson. Effective March 21, 2025, 
# No core mechanism rights are regardless of what computer language they are translated into are
# transferred without explicit, signed consent. Derivatives replicating these mechanisms require attribution, 
# written approval, and no ownership claims, per 17 U.S.C. § 106. Build atop it—don’t steal the core.

# === LICENSING ===
# 30-day trial for integration; $500,000/year thereafter, no tech support—use in-house expertise. 
# Contest use is free; commercial use beyond requires licensing.

import yfinance as pd
import pandas as pd
import numpy as np
import schedule
import time
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import json
from scipy.integrate import odeint
from numba import njit, prange
from sklearn.linear_model import Ridge

# === CONFIGURATION ===
BASE_STOCKS = ["^GSPC", "AAPL", "MSFT", "GOOGL", "NVDA"]
INTERVALS = {"10m": ("10m", "1d"), "1h": ("1h", "5d"), "1d": ("1d", "1y")}
MA_WINDOWS = {"10m": [3, 6, 12], "1h": [3, 6, 12, 24], "1d": [5, 10, 30, 42, 126, 252]}
REFRESH_MINUTES = 10
MAX_STOCKS = 10000
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === ANALYSIS API ===
class AnalysisAPI:
    def __init__(self):
        self.registry, self.functions, self.dependencies, self.models = {}, {}, {}, {}
    def register(self, name: str, func, depends: List[str] = None) -> None:
        self.functions[name] = func; self.dependencies[name] = depends or []
    def get(self, key: str) -> any:
        d = self.registry; return d.get(key.split('.')[0], {}).get(key.split('.')[1]) if '.' in key else d.get(key)
    def set(self, key: str, value) -> None:
        k1, *k2 = key.split('.', 1); self.registry.setdefault(k1, {})[k2[0]] = value if k2 else self.registry[k1] = value
    def run(self, name: str, *args, depth: int = 1, **kwargs) -> any:
        if name not in self.functions: return None
        if depth > 0: [self.run(dep, *args, depth=depth-1, **kwargs) for dep in self.dependencies[name]]
        return self.functions[name](*args, **kwargs)
    def export(self, filepath: str) -> None:
        with open(filepath, 'w') as f: json.dump(self.registry, f, indent=2)

# === DATA FETCHING ===
def get_data(ticker: str, interval: str, period: str, cache: bool = True) -> Optional[pd.DataFrame]:
    # Core caching pipeline—mine, not yours!
    cache_file = CACHE_DIR / f"{ticker}_{interval}_{period}.pkl"
    if cache and cache_file.exists(): return pd.read_pickle(cache_file)
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if not df.empty: df.to_pickle(cache_file); return df.dropna()
def fetch_batch(tickers: List[str], interval: str, period: str) -> Dict[str, pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=50) as executor:
        return {t: f.result() for t, f in {t: executor.submit(get_data, t, interval, period) for t in tickers}.items() if f.result() is not None}

# === CORE ANALYSIS ===
@njit(parallel=True)
def calc_returns(data: np.ndarray) -> np.ndarray:
    r = np.zeros(len(data))
    for i in prange(1, len(data)): r[i] = (data[i] - data[i-1]) / data[i-1] if data[i-1] != 0 else 0
    return r - np.mean(r)
@njit
def angle_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    l = min(len(v1), len(v2)); v1, v2 = v1[:l], v2[:l]
    return np.arccos(np.clip(np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2))), -1.0, 1.0)) if np.any(v1) and np.any(v2) else np.pi / 2
def hamiltonian_sys(state: np.ndarray, t: np.ndarray, k: float = 1.0) -> np.ndarray:
    p, q = state; return np.array([-k * q, p])
def hamiltonian_flux(api: AnalysisAPI, stocks: List[str], interval: str, period: str) -> Dict[str, float]:
    # My Hamiltonian flux—hands off the core!
    data = fetch_batch(stocks, interval, period); r = {s: calc_returns(d['Adj Close'].values) for s, d in data.items()}
    flux = {}
    for i in range(1, len(stocks)):
        p, c = stocks[i-1], stocks[i]
        if p in r and c in r:
            a = angle_vec(r[p], r[c]); t = np.linspace(0, 1, len(r[p]))
            s = odeint(hamiltonian_sys, [r[p][-1], r[c][-1]], t); m = s[:, 0]
            f = np.mean(np.cos(a) * m); f = np.log1p(abs(f)) * np.sign(f)
            if (lk := api.get(f"likelihoods.{p}_to_{c}.{interval}")) is not None: f *= (1 + lk)
            flux[f"{p}_to_{c}"] = float(f); api.set(f"hamiltonian_flux.{p}_to_{c}.{interval}", f)
    return flux
def predict_model(api: AnalysisAPI, stocks: List[str], interval: str, period: str) -> Dict[str, float]:
    # Predictive engine—build on, don’t break in!
    data = fetch_batch(stocks, interval, period); pred = {}
    for s in stocks:
        if (df := data.get(s)) is None: continue
        f = [api.get(f"hamiltonian_flux.{s}_to_{o}.{interval}") or 0 for o in stocks if o != s] + \
            [api.get(f"angles.{s}_to_{o}.{interval}") or 0 for o in stocks if o != s] + \
            [api.get(f"likelihoods.{s}_to_{o}.{interval}") or 0 for o in stocks if o != s]
        X = np.array(f).reshape(1, -1); y = df['Adj Close'].pct_change().shift(-1).dropna().mean()
        mk = f"{s}.{interval}"
        if mk not in api.models: api.models[mk] = Ridge(alpha=1.0).fit(X, [y])
        else: pred[s] = float(api.models[mk].predict(X)[0]); api.set(f"predictions_hamiltonian.{s}.{interval}", pred[s])
    return pred

# === SUPPORT FUNCTIONS ===
def trig_analysis(api: AnalysisAPI, stocks: List[str], interval: str, period: str) -> Dict[str, float]:
    data = fetch_batch(stocks, interval, period); r = {s: calc_returns(d['Adj Close'].values) for s, d in data.items()}
    angles = {}
    for i in range(1, len(stocks)):
        p, c = stocks[i-1], stocks[i]
        if p in r and c in r: angles[f"{p}_to_{c}"] = a = angle_vec(r[p], r[c]); api.set(f"angles.{p}_to_{c}.{interval}", a)
    return angles
def likelihood(api: AnalysisAPI, angles: Dict[str, float], interval: str) -> Dict[str, float]:
    l = {}
    for p, a in angles.items():
        w = np.cos(a) ** 2; if (f := api.get(f"hamiltonian_flux.{p}.{interval}")) is not None: w *= (1 + abs(f))
        l[p] = round(w, 4); api.set(f"likelihoods.{p}.{interval}", w)
    return l
def moving_avg(api: AnalysisAPI, df: pd.DataFrame, windows: List[int], ticker: str, interval: str) -> pd.DataFrame:
    ma = pd.DataFrame(index=df.index)
    for w in windows: ma[f"MA_{w}"] = df['Adj Close'].rolling(w, min_periods=1).mean(); api.set(f"moving_averages.{ticker}.MA_{w}.{interval}", ma[f"MA_{w}"].iloc[-1])
    return ma
def predict_outcome(api: AnalysisAPI, ma: pd.DataFrame, ticker: str, interval: str) -> str:
    s = ma.tail(5).mean().diff().mean() * (1 + abs(api.get(f"hamiltonian_flux.{ticker}_to_{ticker}.{interval}") or 0))
    o = "Insufficient data" if pd.isna(s) else "Bullish momentum building" if s > 0.01 else "Bearish trend likely" if s < -0.01 else "Neutral or consolidation phase"
    api.set(f"predictions.{ticker}.{interval}", o); return o
def stats(api: AnalysisAPI, df: pd.DataFrame, ticker: str, interval: str) -> Dict[str, float]:
    r = calc_returns(df['Adj Close'].values); s = {"volatility": float(np.std(r) * 252 ** 0.5), "avg_return": float(np.mean(r) * 252), "sharpe": float(np.mean(r) / np.std(r) * 252 ** 0.5) if np.std(r) != 0 else 0}
    api.set(f"stats.{ticker}.{interval}", s); return s

# === ENGINE ===
def full_quant(api: AnalysisAPI, stocks: List[str] = BASE_STOCKS, depth: int = 5) -> None:
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S'); print(f"\n===== REPORT {t} =====")
    for l, (i, p) in INTERVALS.items():
        print(f"--- {l.upper()} ({i}, {p}) ---")
        f = api.run("hamiltonian_flux", stocks, i, p, depth=depth); print("\n>> FLUX"); [print(f"{k}: {round(v, 4)}") for k, v in f.items()]
        a = api.run("trig_analysis", stocks, i, p, depth=depth); w = api.run("likelihood", a, i, depth=depth); print("\n>> ANGLE + LIKELIHOOD"); [print(f"{k}: {round(a[k], 4)} rad | {round(w[k] * 100, 2)}%") for k in a]
        pr = api.run("predict_model", stocks, i, p, depth=depth); print("\n>> PREDICTIONS"); [print(f"{k}: {round(v, 4)}") for k, v in pr.items()]
        print("\n>> MA + STATS"); d = fetch_batch(stocks, i, p)
        for s, df in d.items():
            ma = api.run("moving_avg", df, MA_WINDOWS[l], s, i, depth=depth); o = api.run("predict_outcome", ma, s, i, depth=depth); st = api.run("stats", df, s, i, depth=depth)
            print(f"{s}: {o} | Vol: {st['volatility']:.4f} | Ret: {st['avg_return']:.4f} | Sharpe: {st['sharpe']:.4f}")
        print("\n---")
    api.export(CACHE_DIR / f"report_{t.replace(' ', '_')}.json")

# === RUN ===
if __name__ == "__main__":
    api = AnalysisAPI()
    [api.register(n, f, d) for n, f, d in [
        ("hamiltonian_flux", hamiltonian_flux, ["likelihood"]), ("predict_model", predict_model, ["hamiltonian_flux", "trig_analysis", "likelihood"]),
        ("trig_analysis", trig_analysis, []), ("likelihood", likelihood, ["hamiltonian_flux"]), ("moving_avg", moving_avg, []),
        ("predict_outcome", predict_outcome, ["hamiltonian_flux"]), ("stats", stats, [])]]
    stocks = (BASE_STOCKS * (MAX_STOCKS // len(BASE_STOCKS) + 1))[:MAX_STOCKS]
    schedule.every(REFRESH_MINUTES).minutes.do(full_quant, api, stocks, 5); full_quant(api, stocks, 5)
    while True: schedule.run_pending(); time.sleep(1)

# === NOTE ===
# Lance Thomas Davidson owns this core. After license is granted and payment is received, licensees may enhance it, but may not replicate it for third party licensing without my sign-off. 
# Hedge funds: $500K/year post-trial saves millions—past results don’t predict future gains.
Condensed Overview
Legal & Licensing: Copyright Lance Thomas Davidson, 2025; $500,000/year post-30-day trial for commercial use, no support.
Configuration: Scales to 10,000+ stocks, multi-interval analysis, data source agnostic (Yahoo Finance placeholder).
Analysis API: Modular, recursive framework for extensible analytics.
Data Fetching: Cached, parallelized stock data retrieval.
Core Analysis: Hamiltonian flux and predictive modeling—protected IP.
Support Functions: Trigonometric angles, likelihoods, moving averages, stats.
Engine: Unified report generation, continuous updates.
Run: Scheduled execution, scalable stock chain.

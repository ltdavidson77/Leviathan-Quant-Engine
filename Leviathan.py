import yfinance as yf
import pandas as pd
import numpy as np
import schedule
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import json
from scipy.integrate import odeint
from numba import njit, prange
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import requests

BASE_STOCKS = ["^GSPC", "AAPL", "MSFT", "GOOGL", "NVDA"]
INTERVALS = {
    "10m": ("10m", "1d"),
    "1h": ("1h", "5d"),
    "1d": ("1d", "1y")
}
MA_WINDOWS = {
    "10m": [3, 6, 12],
    "1h": [3, 6, 12, 24],
    "1d": [5, 10, 30, 42, 126, 252, 1260, 2520]
}
REFRESH_MINUTES = 10
MAX_STOCKS = 10000
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnalysisAPI:
    def __init__(self):
        self.registry: Dict[str, Dict] = {}
        self.functions: Dict[str, Callable] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.predictive_models: Dict[str, Any] = {}

    def register_function(self, name: str, func: Callable, depends_on: List[str] = None) -> None:
        self.functions[name] = func
        self.dependencies[name] = depends_on or []
        logging.info(f"Registered function: {name} with dependencies: {self.dependencies[name]}")

    def get(self, key: str) -> Any:
        keys = key.split('.')
        current = self.registry
        for k in keys:
            if k not in current:
                return None
            current = current[k]
        return current

    def set(self, key: str, value: Any) -> None:
        keys = key.split('.')
        current = self.registry
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def run_analysis(self, name: str, *args, recursive_depth: int = 1, **kwargs) -> Any:
        if name not in self.functions:
            logging.error(f"Analysis {name} not registered")
            return None
        if recursive_depth > 0 and self.dependencies[name]:
            for dep in self.dependencies[name]:
                self.run_analysis(dep, *args, recursive_depth=recursive_depth-1, **kwargs)
        return self.functions[name](*args, **kwargs)

    def export_to_json(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.registry, f, indent=2)
        logging.info(f"Exported analysis to {filepath}")

class CUDAInferenceEngine:
    def __init__(self, use_native_cuda: bool = False, api_endpoint: str = None, api_key: str = None):
        self.use_native_cuda = use_native_cuda
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def process_unified_data(self, unified_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_native_cuda:
            return self._native_cuda_inference(unified_data)
        elif self.api_endpoint:
            return self._external_api_inference(unified_data)
        else:
            return self._default_processing(unified_data)

    def _native_cuda_inference(self, unified_data: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        for interval, data in unified_data.items():
            processed[interval] = {}
            for stock in data['stocks']:
                combined_features = [
                    data['flux'].get(f"{stock}_to_{stock}", 0),
                    data['angles'].get(f"{stock}_to_{stock}", 0),
                    data['likelihoods'].get(f"{stock}_to_{stock}", 0),
                    data['predictions'].get(stock, 0),
                    *[data['moving_averages'][stock].get(f"MA_{w}", 0) for w in MA_WINDOWS[interval]],
                    data['stats'][stock]['volatility'],
                    data['stats'][stock]['avg_return'],
                    data['stats'][stock]['sharpe']
                ]
                processed[interval][stock] = np.mean(combined_features)
        return processed

    def _external_api_inference(self, unified_data: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = json.dumps(unified_data)
        try:
            response = requests.post(self.api_endpoint, headers=headers, data=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"API inference failed: {e}")
            return self._default_processing(unified_data)

    def _default_processing(self, unified_data: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        for interval, data in unified_data.items():
            processed[interval] = {}
            for stock in data['stocks']:
                combined_score = (
                    data['predictions'].get(stock, 0) +
                    data['stats'][stock]['sharpe'] +
                    data['flux'].get(f"{stock}_to_{stock}", 0)
                ) / 3
                processed[interval][stock] = combined_score
        return processed

def get_data(ticker: str, interval: str, period: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{ticker}_{interval}_{period}.pkl"
    if use_cache and cache_file.exists():
        return pd.read_pickle(cache_file)
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if not df.empty:
            df.to_pickle(cache_file)
            return df.dropna()
        else:
            logging.warning(f"No data for {ticker}")
            return None
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")
        return None

def fetch_batch_data(tickers: List[str], interval: str, period: str) -> Dict[str, pd.DataFrame]:
    data_map = {}
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_ticker = {executor.submit(get_data, ticker, interval, period): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None:
                    data_map[ticker] = df
            except Exception as e:
                logging.error(f"Exception for {ticker}: {e}")
    return data_map

@njit(parallel=True)
def calculate_returns_array(data: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(data))
    for i in prange(1, len(data)):
        returns[i] = (data[i] - data[i-1]) / data[i-1] if data[i-1] != 0 else 0
    returns -= np.mean(returns)
    return returns

def calculate_returns(df: pd.DataFrame) -> np.ndarray:
    return calculate_returns_array(df['Adj Close'].values)

@njit
def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    len_min = min(len(v1), len(v2))
    v1, v2 = v1[:len_min], v2[:len_min]
    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2))
    return np.arccos(np.clip(numerator / denominator, -1.0, 1.0)) if denominator != 0 else np.pi / 2

def hamiltonian_derivative_system(state: np.ndarray, t: np.ndarray, k: float = 1.0) -> np.ndarray:
    p, q = state
    dp_dt = -k * q
    dq_dt = p
    return np.array([dp_dt, dq_dt])

def hamiltonian_flux_analysis(api: AnalysisAPI, stock_chain: List[str], interval: str, period: str) -> Dict[str, float]:
    data_map = fetch_batch_data(stock_chain, interval, period)
    returns_map = {stock: calculate_returns(df) for stock, df in data_map.items()}
    flux_map = {}
    for i in range(1, len(stock_chain)):
        prev, curr = stock_chain[i - 1], stock_chain[i]
        if prev in returns_map and curr in returns_map:
            angle = compute_angle_between_vectors(returns_map[prev], returns_map[curr])
            t = np.linspace(0, 1, len(returns_map[prev]))
            initial_state = [returns_map[prev][-1], returns_map[curr][-1]]
            states = odeint(hamiltonian_derivative_system, initial_state, t)
            momentum = states[:, 0]
            base_flux = np.mean(np.cos(angle) * momentum)
            log_flux = np.log1p(np.abs(base_flux)) * np.sign(base_flux)
            lik_key = f"likelihoods.{prev}_to_{curr}.{interval}"
            if api.get(lik_key) is not None:
                log_flux *= (1 + api.get(lik_key))
            key = f"{prev}_to_{curr}"
            flux_map[key] = float(log_flux)
            api.set(f"hamiltonian_flux.{key}.{interval}", flux_map[key])
    return flux_map

def hamiltonian_predictive_model(api: AnalysisAPI, stock_chain: List[str], interval: str, period: str) -> Dict[str, float]:
    data_map = fetch_batch_data(stock_chain, interval, period)
    predictions = {}
    for stock in stock_chain:
        df = data_map.get(stock)
        if df is None:
            continue
        features = []
        for i, other_stock in enumerate(stock_chain):
            if stock != other_stock:
                flux_key = f"hamiltonian_flux.{stock}_to_{other_stock}.{interval}"
                angle_key = f"angles.{stock}_to_{other_stock}.{interval}"
                lik_key = f"likelihoods.{stock}_to_{other_stock}.{interval}"
                flux = api.get(flux_key) or 0
                angle = api.get(angle_key) or 0
                lik = api.get(l  = api.get(lik_key) or 0
                features.extend([flux, angle, lik])
        X = np.array(features).reshape(1, -1)
        y = df['Adj Close'].pct_change().shift(-1).dropna().mean()
        model_key = f"{stock}.{interval}"
        if model_key not in api.predictive_models:
            api.predictive_models[model_key] = Ridge(alpha=1.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            api.predictive_models[model_key].fit(X_scaled, [y])
        else:
            X_scaled = StandardScaler().fit_transform(X)
        pred = api.predictive_models[model_key].predict(X_scaled)[0]
        predictions[stock] = float(pred)
        api.set(f"predictions_hamiltonian.{stock}.{interval}", pred)
    return predictions

def trigonometric_tier_analysis(api: AnalysisAPI, stock_chain: List[str], interval: str, period: str) -> Dict[str, float]:
    data_map = fetch_batch_data(stock_chain, interval, period)
    returns_map = {stock: calculate_returns(df) for stock, df in data_map.items()}
    angles_map = {}
    for i in range(1, len(stock_chain)):
        prev, curr = stock_chain[i - 1], stock_chain[i]
        if prev in returns_map and curr in returns_map:
            angle = compute_angle_between_vectors(returns_map[prev], returns_map[curr])
            angles_map[f"{prev}_to_{curr}"] = angle
            api.set(f"angles.{prev}_to_{curr}.{interval}", angle)
    return angles_map

def calculate_likelihood_weights(api: AnalysisAPI, angles_map: Dict[str, float], interval: str) -> Dict[str, float]:
    likelihoods = {}
    for pair, angle in angles_map.items():
        weight = np.cos(angle) ** 2
        flux_key = f"hamiltonian_flux.{pair}.{interval}"
        if api.get(flux_key) is not None:
            weight *= (1 + np.abs(api.get(flux_key)))
        likelihoods[pair] = round(weight, 4)
        api.set(f"likelihoods.{pair}.{interval}", weight)
    return likelihoods

def analyze_moving_averages(api: AnalysisAPI, df: pd.DataFrame, windows: List[int], ticker: str, interval: str) -> pd.DataFrame:
    ma_df = pd.DataFrame(index=df.index)
    for w in windows:
        ma_df[f"MA_{w}"] = df['Adj Close'].rolling(window=w, min_periods=1).mean()
        api.set(f"moving_averages.{ticker}.MA_{w}.{interval}", ma_df[f"MA_{w}"].to_dict())
    return ma_df

def predictive_outcome(api: AnalysisAPI, ma_df: pd.DataFrame, ticker: str, interval: str) -> str:
    recent_values = ma_df.tail(5).mean()
    slope = recent_values.diff().mean()
    flux_key = f"hamiltonian_flux.{ticker}_to_{ticker}.{interval}"
    flux = api.get(flux_key) or 0
    slope_adjusted = slope * (1 + np.abs(flux))
    if pd.isna(slope_adjusted):
        outcome = "Insufficient data"
    elif slope_adjusted > 0.01:
        outcome = "Bullish momentum building"
    elif slope_adjusted < -0.01:
        outcome = "Bearish trend likely"
    else:
        outcome = "Neutral or consolidation phase"
    api.set(f"predictions.{ticker}.{interval}", outcome)
    return outcome

def compute_summary_stats(api: AnalysisAPI, df: pd.DataFrame, ticker: str, interval: str) -> Dict[str, float]:
    returns = calculate_returns(df)
    stats = {
        "volatility": float(np.std(returns) * np.sqrt(252)),
        "avg_return": float(np.mean(returns) * 252),
        "sharpe": float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) != 0 else 0
    }
    api.set(f"stats.{ticker}.{interval}", stats)
    return stats

def full_quant_analysis(api: AnalysisAPI, stock_chain: List[str] = BASE_STOCKS, recursive_depth: int = 5):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n===== FULL REPORT {timestamp} =====\n")
    inference_engine = CUDAInferenceEngine(use_native_cuda=False, api_endpoint=None, api_key=None)
    unified_data = {}
    
    for label, (interval, period) in INTERVALS.items():
        print(f"--- Interval: {label.upper()} ({interval}, {period}) ---")
        flux_map = api.run_analysis("hamiltonian_flux_analysis", stock_chain, interval, period, recursive_depth=recursive_depth)
        print("\n>> HAMILTONIAN COSINE FLUX LOGARITHM")
        for pair, flux in flux_map.items():
            print(f"{pair}: {round(flux, 4)} log-flux")
        
        angles = api.run_analysis("trigonometric_tier_analysis", stock_chain, interval, period, recursive_depth=recursive_depth)
        likelihoods = api.run_analysis("calculate_likelihood_weights", angles, interval, recursive_depth=recursive_depth)
        print("\n>> ANGLE + LIKELIHOOD")
        for pair in angles:
            angle = round(angles[pair], 4)
            lik = round(likelihoods[pair] * 100, 2)
            print(f"{pair}: {angle} rad | Likelihood: {lik}%")
        
        predictions = api.run_analysis("hamiltonian_predictive_model", stock_chain, interval, period, recursive_depth=recursive_depth)
        print("\n>> HAMILTONIAN PREDICTIVE MODEL")
        for stock, pred in predictions.items():
            print(f"{stock}: Predicted Return = {round(pred, 4)}")
        
        print("\n>> MOVING AVERAGE SIGNALS & STATS")
        data_map = fetch_batch_data(stock_chain, interval, period)
        ma_data = {}
        stats_data = {}
        for stock, df in data_map.items():
            ma_df = api.run_analysis("analyze_moving_averages", df, MA_WINDOWS[label], stock, interval, recursive_depth=recursive_depth)
            signal = api.run_analysis("predictive_outcome", ma_df, stock, interval, recursive_depth=recursive_depth)
            stats = api.run_analysis("compute_summary_stats", df, stock, interval, recursive_depth=recursive_depth)
            print(f"{stock}: {signal} | Vol: {stats['volatility']:.4f} | Avg Ret: {stats['avg_return']:.4f} | Sharpe: {stats['sharpe']:.4f}")
            ma_data[stock] = {f"MA_{w}": ma_df[f"MA_{w}"].iloc[-1] for w in MA_WINDOWS[label]}
            stats_data[stock] = stats
        
        unified_data[label] = {
            "stocks": stock_chain,
            "flux": flux_map,
            "angles": angles,
            "likelihoods": likelihoods,
            "predictions": predictions,
            "moving_averages": ma_data,
            "stats": stats_data
        }
        print("\n------------------------------------\n")
    
    unified_output = inference_engine.process_unified_data(unified_data)
    print("\n>> UNIFIED AI-ENHANCED OUTPUT")
    for interval, results in unified_output.items():
        print(f"Interval: {interval.upper()}")
        for stock, score in results.items():
            print(f"{stock}: Unified Score = {round(score, 4)}")
    
    api.export_to_json(CACHE_DIR / f"report_{timestamp.replace(' ', '_')}.json")

def expand_stock_chain(base_stocks: List[str], target_size: int = MAX_STOCKS) -> List[str]:
    return (base_stocks * (target_size // len(base_stocks) + 1))[:target_size]

if __name__ == "__main__":
    api = AnalysisAPI()
    api.register_function(
        "hamiltonian_flux_analysis",
        lambda *args, **kwargs: hamiltonian_flux_analysis(api, *args, **kwargs),
        depends_on=["calculate_likelihood_weights"]
    )
    api.register_function(
        "hamiltonian_predictive_model",
        lambda *args, **kwargs: hamiltonian_predictive_model(api, *args, **kwargs),
        depends_on=["hamiltonian_flux_analysis", "trigonometric_tier_analysis", "calculate_likelihood_weights"]
    )
    api.register_function(
        "trigonometric_tier_analysis",
        lambda *args, **kwargs: trigonometric_tier_analysis(api, *args, **kwargs),
        depends_on=[]
    )
    api.register_function(
        "calculate_likelihood_weights",
        lambda *args, **kwargs: calculate_likelihood_weights(api, *args, **kwargs),
        depends_on=["hamiltonian_flux_analysis"]
    )
    api.register_function(
        "analyze_moving_averages",
        lambda *args, **kwargs: analyze_moving_averages(api, *args, **kwargs),
        depends_on=[]
    )
    api.register_function(
        "predictive_outcome",
        lambda *args, **kwargs: predictive_outcome(api, *args, **kwargs),
        depends_on=["hamiltonian_flux_analysis"]
    )
    api.register_function(
        "compute_summary_stats",
        lambda *args, **kwargs: compute_summary_stats(api, *args, **kwargs),
        depends_on=[]
    )

    stock_chain = expand_stock_chain(BASE_STOCKS, MAX_STOCKS)
    schedule.every(REFRESH_MINUTES).minutes.do(full_quant_analysis, api, stock_chain, recursive_depth=5)
    full_quant_analysis(api, stock_chain, recursive_depth=5)
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down analysis engine...")
            api.export_to_json(CACHE_DIR / "final_report.json")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(60)

# README — Legal Terms, Conditions & Intellectual Property Disclaimer
#
# Intellectual Property Notice
#
# This software package, including but not limited to its source code, structural logic, architecture, 
# documentation, and data interactions (hereinafter referred to as “the Software”), is the exclusive 
# intellectual property of its original author(s) and/or legal rights holders. Use, distribution, 
# modification, or integration of the Software, in whole or in part, is subject to the legally binding 
# terms and conditions outlined herein.
#
# Trial Period and Licensing
#
# The Software is delivered as a base model, intended for further refinement, system testing, and 
# integration within various environments. A 30-day trial period is provided for the licensee to 
# evaluate its performance, compatibility, and fitness for purpose. All license payments are final 
# and non-refundable. It is the duty of the licensee to complete all relevant technical and legal 
# due diligence prior to any license purchase.
#
# Environmental Integration & Implementation Responsibility
#
# This Software is distributed as is, without any express or implied warranty. While efforts have 
# been made to maintain its functional and logical integrity, no guarantee is made that the Software 
# will operate consistently across all platforms, infrastructures, or runtime environments. Any 
# adaptation, configuration, deployment, or operational integration—including environmental variable 
# alignment, third-party library interaction, or architecture-dependent performance tuning—remains 
# the sole responsibility of the licensee or implementing party.
#
# Cybersecurity Obligations
#
# All responsibilities related to the Software’s security posture—including but not limited to:
# - Encryption key pair generation and management
# - API key storage and usage
# - Authentication workflows
# - Secure endpoint configuration
# - Output sanitization and log handling
# - Transport and access control measures
# —fall entirely upon the licensee. The authors disclaim all liability for vulnerabilities, data 
# breaches, or system compromises resulting from:
# - Insufficient or improper key management
# - Misconfigured authentication or access controls
# - Exposure through unsecured channels, logs, or response outputs
# - Any form of unintended data disclosure or interaction leakage
#
# Cybersecurity implementation, enforcement, and auditability across production, staging, and 
# sandbox environments shall remain under the exclusive authority and liability of the party 
# deploying or modifying the Software.
#
# Derivation, Reconstitution & Reverse Engineering Prohibition
#
# No portion of the Software may be:
# - Reverse engineered
# - Reconstructed
# - Extrapolated
# - Translated into another programming language
# - Reused in part or in whole as the basis of a separate system
# - Subject to model tracing, workflow extraction, or derivative abstraction
#
# All such activities are strictly prohibited unless formally authorized in writing by the rights 
# holder. This applies to manual, automated, or AI-assisted methods of reverse analysis or 
# transformation.
#
# Whether used together or separately, in conjunction with third-party services or as a standalone 
# module, the Software and any conceptual models derived therefrom are protected under full 
# intellectual property law.
#
# Disclaimer of Liability
#
# The authors, contributors, and rights holders shall not be held liable for any:
# - Direct, indirect, or consequential damages
# - Data loss or corruption
# - Integration failures
# - Platform incompatibilities
# - Security breaches or system exposure
# - Regulatory non-compliance
# - Business interruption, economic loss, or reputational damage
#
# arising from or associated with the use, misuse, or attempted modification of the Software.
#
# Reservation of Rights
#
# All rights are strictly reserved. No use beyond the scope of this license shall be presumed. 
# Unauthorized derivations, translations, commercializations, or redistributions may result in 
# civil and/or criminal penalties under applicable intellectual property and trade secret laws.

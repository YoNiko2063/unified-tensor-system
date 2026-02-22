"""Numeric kernel templates — pure computational Rust + pyo3 wrappers.

Follows rust-physics-kernel/src/lib.rs pattern:
  Layer 1: Pure core (#[inline(always)], stack arrays, no heap in inner loop)
  Layer 2: PyO3 wrapper (#[pyfunction], Vec conversion, PyResult)

All templates have E_borrow < 0.10, well within D_SEP=0.43.
"""

from __future__ import annotations

from typing import Dict

from codegen.intent_spec import BorrowProfile
from codegen.template_registry import RustTemplate, TemplateRegistry


def _sma_render(params: Dict) -> str:
    """Simple Moving Average — pure functional, window-based reduction."""
    window = params.get("window", 20)
    mod_name = params.get("module_name", "sma_kernel")
    return f'''\
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Simple Moving Average — pure core.
#[inline(always)]
pub fn sma(data: &[f64], window: usize) -> Vec<f64> {{
    if data.len() < window || window == 0 {{
        return Vec::new();
    }}
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {{
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }}
    result
}}

/// PyO3 wrapper.
#[pyfunction]
#[pyo3(signature = (data, window={window}))]
fn py_sma(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {{
    if window == 0 {{
        return Err(PyValueError::new_err("window must be > 0"));
    }}
    Ok(sma(&data, window))
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_sma, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_sma_basic() {{
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }}

    #[test]
    fn test_sma_empty() {{
        assert!(sma(&[], 3).is_empty());
        assert!(sma(&[1.0, 2.0], 3).is_empty());
    }}
}}
'''


def _ema_render(params: Dict) -> str:
    """Exponential Moving Average — pure functional, streaming."""
    window = params.get("window", 20)
    mod_name = params.get("module_name", "ema_kernel")
    return f'''\
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Exponential Moving Average — pure core.
/// alpha = 2.0 / (window + 1) as standard smoothing factor.
#[inline(always)]
pub fn ema(data: &[f64], window: usize) -> Vec<f64> {{
    if data.is_empty() || window == 0 {{
        return Vec::new();
    }}
    let alpha = 2.0 / (window as f64 + 1.0);
    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);
    for i in 1..data.len() {{
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }}
    result
}}

#[pyfunction]
#[pyo3(signature = (data, window={window}))]
fn py_ema(data: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {{
    if window == 0 {{
        return Err(PyValueError::new_err("window must be > 0"));
    }}
    Ok(ema(&data, window))
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_ema, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_ema_basic() {{
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 3);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-10);
        // alpha = 0.5 → result[1] = 0.5*2 + 0.5*1 = 1.5
        assert!((result[1] - 1.5).abs() < 1e-10);
    }}

    #[test]
    fn test_ema_empty() {{
        assert!(ema(&[], 3).is_empty());
    }}
}}
'''


def _rsi_render(params: Dict) -> str:
    """Relative Strength Index — pure functional, 0-100 bounded."""
    period = params.get("period", 14)
    mod_name = params.get("module_name", "rsi_kernel")
    return f'''\
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Relative Strength Index — pure core.
/// Uses Wilder's smoothing method.
#[inline(always)]
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {{
    if prices.len() <= period || period == 0 {{
        return Vec::new();
    }}
    let mut result = Vec::with_capacity(prices.len() - period);

    // Initial average gain/loss over first period
    let mut avg_gain = 0.0_f64;
    let mut avg_loss = 0.0_f64;
    for i in 1..=period {{
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {{
            avg_gain += change;
        }} else {{
            avg_loss -= change;
        }}
    }}
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    let rs = if avg_loss.abs() < 1e-15 {{ f64::INFINITY }} else {{ avg_gain / avg_loss }};
    result.push(100.0 - 100.0 / (1.0 + rs));

    // Wilder's smoothing for subsequent values
    let p = period as f64;
    for i in (period + 1)..prices.len() {{
        let change = prices[i] - prices[i - 1];
        let (gain, loss) = if change > 0.0 {{ (change, 0.0) }} else {{ (0.0, -change) }};
        avg_gain = (avg_gain * (p - 1.0) + gain) / p;
        avg_loss = (avg_loss * (p - 1.0) + loss) / p;
        let rs = if avg_loss.abs() < 1e-15 {{ f64::INFINITY }} else {{ avg_gain / avg_loss }};
        result.push(100.0 - 100.0 / (1.0 + rs));
    }}
    result
}}

#[pyfunction]
#[pyo3(signature = (prices, period={period}))]
fn py_rsi(prices: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {{
    if period == 0 {{
        return Err(PyValueError::new_err("period must be > 0"));
    }}
    Ok(rsi(&prices, period))
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_rsi, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_rsi_all_up() {{
        let prices: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = rsi(&prices, 14);
        assert!(!result.is_empty());
        // All gains, no losses → RSI should be 100
        assert!((result[0] - 100.0).abs() < 1e-10);
    }}

    #[test]
    fn test_rsi_insufficient_data() {{
        assert!(rsi(&[1.0, 2.0], 14).is_empty());
    }}
}}
'''


def _macd_render(params: Dict) -> str:
    """MACD — pure functional, dual EMA crossover."""
    fast = params.get("fast_period", 12)
    slow = params.get("slow_period", 26)
    signal = params.get("signal_period", 9)
    mod_name = params.get("module_name", "macd_kernel")
    return f'''\
use pyo3::prelude::*;

/// EMA helper — pure, no allocation beyond output.
#[inline(always)]
fn ema_vec(data: &[f64], window: usize) -> Vec<f64> {{
    if data.is_empty() || window == 0 {{
        return Vec::new();
    }}
    let alpha = 2.0 / (window as f64 + 1.0);
    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);
    for i in 1..data.len() {{
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }}
    result
}}

/// MACD — returns (macd_line, signal_line, histogram) as Vec<[f64; 3]>.
pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<[f64; 3]> {{
    if prices.len() < slow {{
        return Vec::new();
    }}
    let ema_fast = ema_vec(prices, fast);
    let ema_slow = ema_vec(prices, slow);

    // MACD line = fast EMA - slow EMA
    let macd_line: Vec<f64> = ema_fast.iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    // Signal line = EMA of MACD line
    let signal_line = ema_vec(&macd_line, signal);

    // Histogram = MACD - signal
    macd_line.iter()
        .zip(signal_line.iter())
        .map(|(m, s)| [*m, *s, m - s])
        .collect()
}}

#[pyfunction]
#[pyo3(signature = (prices, fast={fast}, slow={slow}, signal={signal}))]
fn py_macd(prices: Vec<f64>, fast: usize, slow: usize, signal: usize) -> PyResult<Vec<Vec<f64>>> {{
    Ok(macd(&prices, fast, slow, signal).iter().map(|r| r.to_vec()).collect())
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_macd, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_macd_basic() {{
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let result = macd(&prices, {fast}, {slow}, {signal});
        assert_eq!(result.len(), prices.len());
    }}

    #[test]
    fn test_macd_insufficient() {{
        assert!(macd(&[1.0; 10], {fast}, {slow}, {signal}).is_empty());
    }}
}}
'''


def _bollinger_render(params: Dict) -> str:
    """Bollinger Bands — pure functional, rolling stats."""
    window = params.get("window", 20)
    num_std = params.get("num_std", 2.0)
    mod_name = params.get("module_name", "bollinger_kernel")
    return f'''\
use pyo3::prelude::*;

/// Bollinger Bands — returns (middle, upper, lower) as Vec<[f64; 3]>.
/// Middle = SMA(window), Upper/Lower = Middle ± num_std × σ.
pub fn bollinger(data: &[f64], window: usize, num_std: f64) -> Vec<[f64; 3]> {{
    if data.len() < window || window == 0 {{
        return Vec::new();
    }}
    let mut result = Vec::with_capacity(data.len() - window + 1);

    for i in 0..=(data.len() - window) {{
        let slice = &data[i..i + window];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>() / window as f64;
        let std = variance.sqrt();
        result.push([mean, mean + num_std * std, mean - num_std * std]);
    }}
    result
}}

#[pyfunction]
#[pyo3(signature = (data, window={window}, num_std={num_std}))]
fn py_bollinger(data: Vec<f64>, window: usize, num_std: f64) -> PyResult<Vec<Vec<f64>>> {{
    Ok(bollinger(&data, window, num_std).iter().map(|r| r.to_vec()).collect())
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_bollinger, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_bollinger_constant() {{
        let data = vec![50.0; 30];
        let result = bollinger(&data, 20, 2.0);
        assert_eq!(result.len(), 11);
        // Constant input → std=0 → all bands equal
        for r in &result {{
            assert!((r[0] - 50.0).abs() < 1e-10);
            assert!((r[1] - 50.0).abs() < 1e-10);
            assert!((r[2] - 50.0).abs() < 1e-10);
        }}
    }}
}}
'''


# ── Template registration ────────────────────────────────────────────────────

SMA_TEMPLATE = RustTemplate(
    name="sma",
    domain="numeric",
    operation="sma",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_sma_render,
    description="Simple Moving Average — sliding window reduction",
)

EMA_TEMPLATE = RustTemplate(
    name="ema",
    domain="numeric",
    operation="ema",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_ema_render,
    description="Exponential Moving Average — streaming alpha-blended",
)

RSI_TEMPLATE = RustTemplate(
    name="rsi",
    domain="numeric",
    operation="rsi",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_rsi_render,
    description="Relative Strength Index — Wilder's smoothing",
)

MACD_TEMPLATE = RustTemplate(
    name="macd",
    domain="numeric",
    operation="macd",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_macd_render,
    description="MACD — dual EMA crossover with signal + histogram",
)

BOLLINGER_TEMPLATE = RustTemplate(
    name="bollinger",
    domain="numeric",
    operation="bollinger",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_bollinger_render,
    description="Bollinger Bands — rolling mean ± k*std",
)

ALL_NUMERIC_TEMPLATES = [
    SMA_TEMPLATE,
    EMA_TEMPLATE,
    RSI_TEMPLATE,
    MACD_TEMPLATE,
    BOLLINGER_TEMPLATE,
]


def register_all(registry: TemplateRegistry) -> None:
    """Register all numeric kernel templates into a registry."""
    for t in ALL_NUMERIC_TEMPLATES:
        registry.register(t)

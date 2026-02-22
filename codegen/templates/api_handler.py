"""API handler templates — async HTTP fetch with reqwest + serde.

These templates require Cargo (not standalone rustc) because they depend on
external crates: reqwest, serde, serde_json, tokio.

BorrowProfile: ASYNC_IO → BV=(0.30, 0.10, 0.10, 0.00, 0.00, 0.00), E=0.12
"""

from __future__ import annotations

from typing import Dict

from codegen.intent_spec import BorrowProfile
from codegen.template_registry import RustTemplate, TemplateRegistry


def _alpaca_ohlcv_cargo_toml(params: Dict) -> str:
    """Generate Cargo.toml for Alpaca OHLCV fetch."""
    name = params.get("crate_name", "alpaca_ohlcv")
    return f'''\
[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[dependencies]
reqwest = {{ version = "0.11", features = ["json", "blocking"] }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
pyo3 = {{ version = "0.20", features = ["extension-module"] }}

[lib]
name = "{name}"
crate-type = ["cdylib"]
'''


def _alpaca_ohlcv_render(params: Dict) -> str:
    """Alpaca REST OHLCV fetch — blocking reqwest + serde deserialization."""
    mod_name = params.get("module_name", "alpaca_ohlcv")
    return f'''\
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use serde::Deserialize;

/// Single OHLCV bar from Alpaca API.
#[derive(Deserialize, Debug, Clone)]
pub struct Bar {{
    pub t: String,   // timestamp
    pub o: f64,      // open
    pub h: f64,      // high
    pub l: f64,      // low
    pub c: f64,      // close
    pub v: u64,      // volume
}}

/// Response wrapper for Alpaca bars endpoint.
#[derive(Deserialize, Debug)]
struct BarsResponse {{
    bars: Vec<Bar>,
}}

/// Fetch OHLCV bars from Alpaca Data API (blocking).
///
/// Uses APCA-API-KEY-ID and APCA-API-SECRET-KEY headers.
pub fn fetch_bars(
    symbol: &str,
    timeframe: &str,
    start: &str,
    end: &str,
    api_key: &str,
    api_secret: &str,
) -> Result<Vec<Bar>, String> {{
    let url = format!(
        "https://data.alpaca.markets/v2/stocks/{{}}/bars?timeframe={{}}&start={{}}&end={{}}",
        symbol, timeframe, start, end,
    );

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("APCA-API-KEY-ID", api_key)
        .header("APCA-API-SECRET-KEY", api_secret)
        .send()
        .map_err(|e| format!("Request failed: {{}}", e))?;

    if !resp.status().is_success() {{
        return Err(format!("HTTP {{}}: {{}}", resp.status(), resp.text().unwrap_or_default()));
    }}

    let body: BarsResponse = resp.json()
        .map_err(|e| format!("JSON parse failed: {{}}", e))?;

    Ok(body.bars)
}}

// ── PyO3 bindings ────────────────────────────────────────────────────────────

#[pyfunction]
fn py_fetch_bars(
    symbol: &str,
    timeframe: &str,
    start: &str,
    end: &str,
    api_key: &str,
    api_secret: &str,
) -> PyResult<Vec<Vec<f64>>> {{
    let bars = fetch_bars(symbol, timeframe, start, end, api_key, api_secret)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Return as [[open, high, low, close, volume], ...]
    Ok(bars.iter().map(|b| vec![b.o, b.h, b.l, b.c, b.v as f64]).collect())
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_fetch_bars, m)?)?;
    Ok(())
}}
'''


# ── Template registration ────────────────────────────────────────────────────

ALPACA_OHLCV_TEMPLATE = RustTemplate(
    name="alpaca_ohlcv_fetch",
    domain="api",
    operation="alpaca_ohlcv_fetch",
    borrow_profile=BorrowProfile.ASYNC_IO,
    design_bv=(0.30, 0.10, 0.10, 0.00, 0.00, 0.00),
    render=_alpaca_ohlcv_render,
    requires_cargo=True,
    crate_deps={
        "reqwest": '{ version = "0.11", features = ["json", "blocking"] }',
        "serde": '{ version = "1", features = ["derive"] }',
        "serde_json": '"1"',
        "pyo3": '{ version = "0.20", features = ["extension-module"] }',
    },
    description="Alpaca REST OHLCV fetch — blocking reqwest + serde",
)

ALL_API_TEMPLATES = [ALPACA_OHLCV_TEMPLATE]


def cargo_toml_for(template_name: str, params: Dict) -> str:
    """Generate Cargo.toml for a crate-dependent template."""
    if template_name == "alpaca_ohlcv_fetch":
        return _alpaca_ohlcv_cargo_toml(params)
    raise ValueError(f"No Cargo.toml generator for template: {{template_name}}")


def register_all(registry: TemplateRegistry) -> None:
    """Register all API handler templates into a registry."""
    for t in ALL_API_TEMPLATES:
        registry.register(t)

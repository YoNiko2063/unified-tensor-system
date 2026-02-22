"""Text parser templates — pure functional Rust for text extraction.

Templates:
  ticker_extractor:      Regex-based ticker symbol extraction (BV=0.025)
  article_feature_parser: HTML → structured features (BV=0.10)

Both are pure_functional — no mutable state, no lifetimes.
"""

from __future__ import annotations

from typing import Dict

from codegen.intent_spec import BorrowProfile
from codegen.template_registry import RustTemplate, TemplateRegistry


def _ticker_extractor_render(params: Dict) -> str:
    """Ticker extraction — pure regex, no external crates."""
    mod_name = params.get("module_name", "ticker_extractor")
    return f'''\
use pyo3::prelude::*;

/// Known stop words that look like tickers but aren't.
const STOP_WORDS: &[&str] = &[
    "A", "I", "AM", "PM", "CEO", "CFO", "CTO", "COO", "US", "UK", "EU",
    "GDP", "CPI", "SEC", "FDA", "IPO", "ETF", "NYSE", "API", "LLC", "INC",
    "THE", "AND", "FOR", "ARE", "NOT", "BUT", "HAS", "HAD", "HIS", "HER",
    "ALL", "CAN", "NEW", "OLD", "OUR", "OUT", "OWN", "SAY", "SHE", "TOO",
    "USE", "WAY", "WHO", "BOY", "DID", "GET", "HIM", "HOW", "MAN", "MAY",
    "WAS", "DAY", "LOW", "EST", "ITS", "TWO", "ANY", "FEW", "GOT", "LET",
    "PUT", "RAN", "RED", "RUN", "SET", "SIT", "TOP", "WON", "YET",
];

/// Extract ticker symbols from text.
///
/// Rules:
/// - 1-5 uppercase letters, optionally prefixed with $
/// - Not preceded or followed by word characters
/// - Filtered against STOP_WORDS
///
/// Returns deduplicated list of tickers in order of first appearance.
pub fn extract_tickers(text: &str) -> Vec<String> {{
    let mut tickers = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {{
        // Check for $ prefix or start of uppercase run
        let has_dollar = bytes[i] == b'$';
        let start = if has_dollar {{ i + 1 }} else {{ i }};

        if start >= len || !bytes[start].is_ascii_uppercase() {{
            i += 1;
            continue;
        }}

        // Check preceding character (must not be alphanumeric)
        if !has_dollar && i > 0 && bytes[i - 1].is_ascii_alphanumeric() {{
            i += 1;
            continue;
        }}

        // Scan uppercase run
        let mut end = start;
        while end < len && bytes[end].is_ascii_uppercase() {{
            end += 1;
        }}
        let ticker_len = end - start;

        // Check following character (must not be alphanumeric)
        if end < len && bytes[end].is_ascii_alphanumeric() {{
            i = end;
            continue;
        }}

        // Length filter: 1-5 chars
        if ticker_len >= 1 && ticker_len <= 5 {{
            let ticker = &text[start..end];
            if !STOP_WORDS.contains(&ticker) && seen.insert(ticker.to_string()) {{
                tickers.push(ticker.to_string());
            }}
        }}

        i = end;
    }}
    tickers
}}

#[pyfunction]
fn py_extract_tickers(text: &str) -> Vec<String> {{
    extract_tickers(text)
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_extract_tickers, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_basic_extraction() {{
        let text = "Buy $AAPL and MSFT, sell TSLA. The CEO said GDP is up.";
        let tickers = extract_tickers(text);
        assert!(tickers.contains(&"AAPL".to_string()));
        assert!(tickers.contains(&"MSFT".to_string()));
        assert!(tickers.contains(&"TSLA".to_string()));
        assert!(!tickers.contains(&"CEO".to_string()));
        assert!(!tickers.contains(&"GDP".to_string()));
    }}

    #[test]
    fn test_deduplication() {{
        let text = "$AAPL up then $AAPL down";
        let tickers = extract_tickers(text);
        assert_eq!(tickers.len(), 1);
    }}
}}
'''


def _article_feature_parser_render(params: Dict) -> str:
    """Article feature extraction — HTML → structured features."""
    mod_name = params.get("module_name", "article_features")
    return f'''\
use pyo3::prelude::*;

/// Article features extracted from text content.
pub struct ArticleFeatures {{
    pub word_count: usize,
    pub sentence_count: usize,
    pub avg_sentence_length: f64,
    pub uppercase_ratio: f64,
    pub numeric_ratio: f64,
    pub question_count: usize,
    pub exclamation_count: usize,
}}

/// Strip HTML tags from content (simple state machine).
pub fn strip_html(html: &str) -> String {{
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {{
        match ch {{
            '<' => in_tag = true,
            '>' => {{
                in_tag = false;
                result.push(' ');
            }},
            _ if !in_tag => result.push(ch),
            _ => {{}},
        }}
    }}
    result
}}

/// Extract features from plain text.
pub fn extract_features(text: &str) -> ArticleFeatures {{
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len();

    // Sentence splitting (simple: split on .!?)
    let sentence_count = text.chars()
        .filter(|c| *c == '.' || *c == '!' || *c == '?')
        .count()
        .max(1);

    let avg_sentence_length = word_count as f64 / sentence_count as f64;

    let total_chars = text.len().max(1) as f64;
    let uppercase_count = text.chars().filter(|c| c.is_ascii_uppercase()).count();
    let numeric_count = text.chars().filter(|c| c.is_ascii_digit()).count();

    ArticleFeatures {{
        word_count,
        sentence_count,
        avg_sentence_length,
        uppercase_ratio: uppercase_count as f64 / total_chars,
        numeric_ratio: numeric_count as f64 / total_chars,
        question_count: text.chars().filter(|c| *c == '?').count(),
        exclamation_count: text.chars().filter(|c| *c == '!').count(),
    }}
}}

/// Parse HTML article and extract features.
pub fn parse_article(html: &str) -> ArticleFeatures {{
    let text = strip_html(html);
    extract_features(&text)
}}

#[pyfunction]
fn py_parse_article(html: &str) -> PyResult<Vec<f64>> {{
    let f = parse_article(html);
    Ok(vec![
        f.word_count as f64,
        f.sentence_count as f64,
        f.avg_sentence_length,
        f.uppercase_ratio,
        f.numeric_ratio,
        f.question_count as f64,
        f.exclamation_count as f64,
    ])
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_parse_article, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_strip_html() {{
        let html = "<p>Hello <b>world</b></p>";
        let text = strip_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<"));
    }}

    #[test]
    fn test_features() {{
        let text = "The price went up. Revenue grew 10%. Is this bullish?";
        let f = extract_features(text);
        assert_eq!(f.word_count, 10);
        assert!(f.sentence_count >= 3);
        assert!(f.question_count == 1);
    }}
}}
'''


# ── Template registration ────────────────────────────────────────────────────

TICKER_EXTRACTOR_TEMPLATE = RustTemplate(
    name="ticker_extractor",
    domain="text",
    operation="ticker_extractor",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_ticker_extractor_render,
    description="Regex-based ticker symbol extraction — pure functional",
)

ARTICLE_FEATURE_PARSER_TEMPLATE = RustTemplate(
    name="article_feature_parser",
    domain="text",
    operation="article_feature_parser",
    borrow_profile=BorrowProfile.SHARED_REFERENCE,
    design_bv=(0.20, 0.15, 0.00, 0.00, 0.00, 0.00),
    render=_article_feature_parser_render,
    description="HTML → structured features — shared reference pattern",
)

ALL_TEXT_TEMPLATES = [
    TICKER_EXTRACTOR_TEMPLATE,
    ARTICLE_FEATURE_PARSER_TEMPLATE,
]


def register_all(registry: TemplateRegistry) -> None:
    """Register all text parser templates into a registry."""
    for t in ALL_TEXT_TEMPLATES:
        registry.register(t)

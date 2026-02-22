"""HTML navigator templates — pure Rust HTML parsing without external crates.

Follows rust-html-parser/nom pattern: manual &str slicing instead of
external crates. All templates have E_borrow < D_SEP=0.43.

BorrowVector components (B1..B6):
  B1 = lifetime parameters / &T in function signatures
  B2 = lifetime-annotated references (&'a T)
  B3 = Box/Rc/Arc (heap indirection)
  B4 = &mut T in type positions
  B5 = raw pointers (*const/*mut)
  B6 = &mut <expr> in expression positions
"""

from __future__ import annotations

from typing import Dict

from codegen.intent_spec import BorrowProfile
from codegen.template_registry import RustTemplate, TemplateRegistry


def link_extractor(params: Dict) -> str:
    """Extract all <a href="..."> links from an HTML string.

    Pure functional, owned values, no lifetime annotations.
    BV = (0.10, 0.00, 0.00, 0.00, 0.00, 0.00), E ≈ 0.025
    """
    return '''\
/// Extract all href attribute values from anchor tags in an HTML string.
///
/// Scans the input for `href="..."` patterns (also single-quoted).
/// Returns owned Vec<String> — no borrows, pure functional.
pub fn extract_links(html: String) -> Vec<String> {
    let mut links = Vec::new();
    let bytes = html.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        // Search for "href"
        if i + 4 <= len && &bytes[i..i + 4] == b"href" {
            i += 4;
            // Skip whitespace and '='
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\\t') {
                i += 1;
            }
            if i < len && bytes[i] == b'=' {
                i += 1;
            }
            while i < len && (bytes[i] == b' ' || bytes[i] == b'\\t') {
                i += 1;
            }
            // Expect opening quote
            if i < len && (bytes[i] == b'"' || bytes[i] == b'\\'' ) {
                let quote = bytes[i];
                i += 1;
                let start = i;
                while i < len && bytes[i] != quote {
                    i += 1;
                }
                if i <= len {
                    let href = String::from_utf8_lossy(&bytes[start..i]).into_owned();
                    links.push(href);
                    i += 1; // skip closing quote
                }
            }
        } else {
            i += 1;
        }
    }
    links
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_link() {
        let html = String::from(r#"<a href="https://example.com">click</a>"#);
        let links = extract_links(html);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0], "https://example.com");
    }

    #[test]
    fn test_multiple_links() {
        let html = String::from(
            r#"<a href="https://a.com">A</a> <a href="https://b.com">B</a>"#
        );
        let links = extract_links(html);
        assert_eq!(links.len(), 2);
    }

    #[test]
    fn test_empty_html() {
        let links = extract_links(String::from("<p>no links here</p>"));
        assert!(links.is_empty());
    }

    #[test]
    fn test_single_quoted_href() {
        let html = String::from("<a href=\\'https://single.com\\'>text</a>");
        let links = extract_links(html);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0], "https://single.com");
    }
}
'''


def text_content_extractor(params: Dict) -> str:
    """Strip all HTML tags from input, returning plain text.

    Shared reference to input string slice, no mutation.
    BV = (0.20, 0.10, 0.00, 0.00, 0.00, 0.00), E ≈ 0.068
    """
    preserve_whitespace = params.get("preserve_whitespace", False)
    preserve_str = "true" if preserve_whitespace else "false"
    return f'''\
/// Strip all HTML tags from the input, returning plain text.
///
/// Takes a shared reference to the HTML string.
/// When `preserve_whitespace` is false, collapses runs of whitespace to single space.
pub fn extract_text(html: &str, preserve_whitespace: bool) -> String {{
    let mut output = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut in_tag = false;

    while i < len {{
        if bytes[i] == b\'<\' {{
            in_tag = true;
            i += 1;
        }} else if bytes[i] == b\'>\' {{
            in_tag = false;
            // Insert a space where the tag was, to separate adjacent words
            if !output.ends_with(\' \') {{
                output.push(\' \');
            }}
            i += 1;
        }} else if !in_tag {{
            output.push(bytes[i] as char);
            i += 1;
        }} else {{
            i += 1;
        }}
    }}

    if preserve_whitespace {{
        output
    }} else {{
        // Collapse whitespace runs to single space
        let mut collapsed = String::with_capacity(output.len());
        let mut last_was_space = false;
        for ch in output.chars() {{
            if ch.is_ascii_whitespace() {{
                if !last_was_space {{
                    collapsed.push(\' \');
                }}
                last_was_space = true;
            }} else {{
                collapsed.push(ch);
                last_was_space = false;
            }}
        }}
        collapsed.trim().to_owned()
    }}
}}

/// Convenience: extract text with default parameter ({preserve_str}).
pub fn extract_text_default(html: &str) -> String {{
    extract_text(html, {preserve_str})
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_strip_basic_tags() {{
        let html = "<p>Hello <b>world</b></p>";
        let text = extract_text(html, false);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains('<'));
        assert!(!text.contains('>'));
    }}

    #[test]
    fn test_empty_html() {{
        assert_eq!(extract_text("", false), "");
    }}

    #[test]
    fn test_no_tags() {{
        let text = extract_text("just plain text", false);
        assert_eq!(text, "just plain text");
    }}

    #[test]
    fn test_preserve_whitespace() {{
        let html = "<p>  spaced  </p>";
        let preserved = extract_text(html, true);
        let collapsed = extract_text(html, false);
        // Preserved version has more spaces
        assert!(preserved.len() >= collapsed.len());
    }}
}}
'''


def structured_table_extractor(params: Dict) -> str:
    """Extract <table> rows as Vec<Vec<String>>.

    Mutable output buffer, immutable input reference.
    BV = (0.20, 0.00, 0.00, 0.30, 0.00, 0.00), E ≈ 0.101
    """
    max_rows = params.get("max_rows", 1000)
    header_row = params.get("header_row", True)
    header_str = "true" if header_row else "false"
    return f'''\
/// Extract table rows from HTML as Vec<Vec<String>>.
///
/// Parses <tr> and <td>/<th> tags from the first <table> found.
/// Uses a mutable output buffer; reads from an immutable &str reference.
pub fn extract_table(
    html: &str,
    max_rows: usize,
    include_header_row: bool,
) -> Vec<Vec<String>> {{
    let mut rows: Vec<Vec<String>> = Vec::new();
    let bytes = html.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    // Helper: find closing '>' from current position
    let skip_tag = |start: usize| -> usize {{
        let mut j = start;
        while j < len && bytes[j] != b\'>\' {{
            j += 1;
        }}
        j + 1
    }};

    // Helper: extract text content until '<'
    let extract_cell_text = |start: usize| -> (String, usize) {{
        let mut j = start;
        let mut text = String::new();
        while j < len && bytes[j] != b\'<\' {{
            text.push(bytes[j] as char);
            j += 1;
        }}
        (text.trim().to_owned(), j)
    }};

    let mut in_table = false;
    let mut current_row: Vec<String> = Vec::new();
    let mut first_row_seen = false;

    while i < len && rows.len() < max_rows {{
        if bytes[i] == b\'<\' {{
            // Peek at tag name
            let tag_start = i + 1;
            let mut tag_end = tag_start;
            while tag_end < len && bytes[tag_end] != b\'>\' && !bytes[tag_end].is_ascii_whitespace() {{
                tag_end += 1;
            }}
            let tag_bytes = &bytes[tag_start..tag_end];

            match tag_bytes {{
                b"table" | b"TABLE" => {{
                    in_table = true;
                    i = skip_tag(i + 1);
                }}
                b"/table" | b"/TABLE" => {{
                    if !current_row.is_empty() {{
                        rows.push(current_row.clone());
                        current_row.clear();
                    }}
                    in_table = false;
                    i = skip_tag(i + 1);
                }}
                b"tr" | b"TR" if in_table => {{
                    if !current_row.is_empty() {{
                        if !first_row_seen || include_header_row {{
                            rows.push(current_row.clone());
                        }}
                        first_row_seen = true;
                        current_row.clear();
                    }}
                    i = skip_tag(i + 1);
                }}
                b"/tr" | b"/TR" if in_table => {{
                    if !current_row.is_empty() {{
                        if !first_row_seen || include_header_row {{
                            rows.push(current_row.clone());
                        }}
                        first_row_seen = true;
                        current_row.clear();
                    }}
                    i = skip_tag(i + 1);
                }}
                b"td" | b"TD" | b"th" | b"TH" if in_table => {{
                    i = skip_tag(i + 1);
                    let (cell_text, next_i) = extract_cell_text(i);
                    current_row.push(cell_text);
                    i = next_i;
                }}
                _ => {{
                    i = skip_tag(i + 1);
                }}
            }}
        }} else {{
            i += 1;
        }}
    }}

    rows
}}

/// Convenience with defaults: max_rows={max_rows}, header_row={header_str}.
pub fn extract_table_default(html: &str) -> Vec<Vec<String>> {{
    extract_table(html, {max_rows}, {header_str})
}}

#[cfg(test)]
mod tests {{
    use super::*;

    const SAMPLE_TABLE: &str = r#"
<table>
  <tr><th>Name</th><th>Value</th></tr>
  <tr><td>Alpha</td><td>1</td></tr>
  <tr><td>Beta</td><td>2</td></tr>
</table>
"#;

    #[test]
    fn test_basic_table() {{
        let rows = extract_table(SAMPLE_TABLE, 100, true);
        assert!(!rows.is_empty());
        // Should have 3 rows: header + 2 data
        assert_eq!(rows.len(), 3);
    }}

    #[test]
    fn test_max_rows_limit() {{
        let rows = extract_table(SAMPLE_TABLE, 1, true);
        assert!(rows.len() <= 1);
    }}

    #[test]
    fn test_empty_html() {{
        let rows = extract_table("", 100, true);
        assert!(rows.is_empty());
    }}

    #[test]
    fn test_no_table() {{
        let rows = extract_table("<p>No table here</p>", 100, true);
        assert!(rows.is_empty());
    }}
}}
'''


def url_canonicalizer(params: Dict) -> str:
    """Normalize URLs: strip utm_* params, lowercase scheme/host.

    Pure functional string transformation, owned values.
    BV = (0.10, 0.00, 0.00, 0.00, 0.00, 0.00), E ≈ 0.025
    """
    strip_fragments = params.get("strip_fragments", True)
    strip_str = "true" if strip_fragments else "false"
    return f'''\
/// Canonicalize a URL: lowercase scheme and host, strip utm_* query params.
///
/// Pure functional — takes owned String, returns owned String.
/// No external crates: manual string scanning.
pub fn canonicalize_url(url: String, strip_fragments: bool) -> String {{
    // Split on '?' to separate path from query
    let (base, query_and_frag) = if let Some(pos) = url.find(\'?\') {{
        (&url[..pos], Some(&url[pos + 1..]))
    }} else {{
        (url.as_str(), None)
    }};

    // Separate fragment from query
    let (query_part, fragment_part) = match query_and_frag {{
        None => {{
            // Check if base itself has a fragment
            if let Some(fpos) = base.find(\'#\') {{
                let b = &url[..fpos];
                let f = &url[fpos..];
                return canonicalize_url(
                    format!("{{}}{{}}",
                        lowercase_scheme_host(b),
                        if strip_fragments {{ String::new() }} else {{ f.to_owned() }}
                    ),
                    strip_fragments,
                );
            }}
            (None, None)
        }}
        Some(qf) => {{
            if let Some(fpos) = qf.find(\'#\') {{
                (Some(&qf[..fpos]), Some(&qf[fpos..]))
            }} else {{
                (Some(qf), None)
            }}
        }}
    }};

    // Lowercase scheme + host portion
    let canonical_base = lowercase_scheme_host(base);

    // Filter query params: remove utm_* keys
    let filtered_query = match query_part {{
        None => String::new(),
        Some(q) => {{
            let kept: Vec<&str> = q.split(\'&\')
                .filter(|param| {{
                    let key = param.split(\'=\').next().unwrap_or("");
                    !key.starts_with("utm_") && !key.is_empty()
                }})
                .collect();
            if kept.is_empty() {{
                String::new()
            }} else {{
                format!("?{{}}", kept.join("&"))
            }}
        }}
    }};

    let frag = if strip_fragments {{
        String::new()
    }} else {{
        fragment_part.map(|f| f.to_owned()).unwrap_or_default()
    }};

    format!("{{}}{{}}{{}}", canonical_base, filtered_query, frag)
}}

/// Lowercase the scheme (e.g. HTTPS://) and host portion of a URL.
fn lowercase_scheme_host(url: &str) -> String {{
    // Find "://"
    if let Some(scheme_end) = url.find("://") {{
        let scheme = &url[..scheme_end];
        let rest = &url[scheme_end + 3..];
        // Find end of host (next '/' or end of string)
        let host_end = rest.find(\'/\').unwrap_or(rest.len());
        let host = &rest[..host_end];
        let path = &rest[host_end..];
        format!("{{}}://{{}}{{}}",
            scheme.to_ascii_lowercase(),
            host.to_ascii_lowercase(),
            path)
    }} else {{
        url.to_owned()
    }}
}}

/// Convenience with default strip_fragments={strip_str}.
pub fn canonicalize_url_default(url: String) -> String {{
    canonicalize_url(url, {strip_str})
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_lowercase_scheme() {{
        let url = String::from("HTTPS://Example.COM/path");
        let canon = canonicalize_url(url, true);
        assert!(canon.starts_with("https://example.com"));
    }}

    #[test]
    fn test_strip_utm_params() {{
        let url = String::from("https://example.com/page?id=1&utm_source=google&utm_medium=cpc");
        let canon = canonicalize_url(url, false);
        assert!(canon.contains("id=1"));
        assert!(!canon.contains("utm_source"));
        assert!(!canon.contains("utm_medium"));
    }}

    #[test]
    fn test_strip_fragments() {{
        let url = String::from("https://example.com/page#section");
        let with_frag = canonicalize_url(url.clone(), false);
        let without_frag = canonicalize_url(url, true);
        assert!(with_frag.contains('#'));
        assert!(!without_frag.contains('#'));
    }}

    #[test]
    fn test_clean_url_unchanged() {{
        let url = String::from("https://example.com/page?q=rust");
        let canon = canonicalize_url(url.clone(), true);
        assert!(canon.contains("q=rust"));
        assert!(!canon.contains("utm_"));
    }}
}}
'''


# ── Template objects ─────────────────────────────────────────────────────────

LINK_EXTRACTOR_TEMPLATE = RustTemplate(
    name="link_extractor",
    domain="html",
    operation="link_extractor",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=link_extractor,
    description="Extract all <a href=...> links from HTML string — owned, no borrows",
)

TEXT_CONTENT_EXTRACTOR_TEMPLATE = RustTemplate(
    name="text_content_extractor",
    domain="html",
    operation="text_content_extractor",
    borrow_profile=BorrowProfile.SHARED_REFERENCE,
    design_bv=(0.20, 0.10, 0.00, 0.00, 0.00, 0.00),
    render=text_content_extractor,
    description="Strip HTML tags and return plain text — shared &str reference",
)

STRUCTURED_TABLE_EXTRACTOR_TEMPLATE = RustTemplate(
    name="structured_table_extractor",
    domain="html",
    operation="structured_table_extractor",
    borrow_profile=BorrowProfile.MUTABLE_OUTPUT,
    design_bv=(0.20, 0.00, 0.00, 0.30, 0.00, 0.00),
    render=structured_table_extractor,
    description="Extract <table> rows as Vec<Vec<String>> — mutable output buffer",
)

URL_CANONICALIZER_TEMPLATE = RustTemplate(
    name="url_canonicalizer",
    domain="html",
    operation="url_canonicalizer",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=url_canonicalizer,
    description="Normalize URL: lowercase scheme/host, strip utm_* params",
)

ALL_HTML_TEMPLATES = [
    LINK_EXTRACTOR_TEMPLATE,
    TEXT_CONTENT_EXTRACTOR_TEMPLATE,
    STRUCTURED_TABLE_EXTRACTOR_TEMPLATE,
    URL_CANONICALIZER_TEMPLATE,
]


def register_all(registry: TemplateRegistry) -> None:
    """Register all HTML navigator templates into a registry."""
    for t in ALL_HTML_TEMPLATES:
        registry.register(t)

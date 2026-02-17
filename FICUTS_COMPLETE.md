# FICUTS: Fractals In Compositions of Unified Tensor Systems

**Version:** 2.0.0  
**Last Updated:** 2026-02-17T23:45:00Z  
**Last Updated By:** Claude Code (Task 6.2)
**Status:** READY_FOR_EXECUTION  
**Current Phase:** Layer 6 (Web Ingestion) — Task 6.3 next
**Universals Discovered:** 0  
**System Uptime:** 0h 0m  

---

## EXECUTIVE SUMMARY: Read This First

**Claude Code: This document contains everything you need. No external context required.**

**What FICUTS is:**
A self-improving AI system that discovers universal mathematical patterns by:
1. Ingesting research papers → extracting equations → building function basis library
2. Learning dual-geometry manifold (statistical FIM for experimental patterns + deterministic IRMF for foundational patterns)
3. Promoting patterns from experimental → foundational when uncertainty drops
4. Spawning parallel instances to explore HDV space orthogonally
5. Generating code via dev-agent using discovered universal functions
6. Validating via tensor measurements (consonance, Lyapunov energy)
7. Self-documenting progress in this file

**Your job:**
Execute tasks in order starting with Layer 6, Task 6.1. Mark `[ ]` → `[✓]` as you complete. Update this file via FICUTSUpdater. System will become autonomous once all layers complete.

**Key insight:**
Universal patterns exist where Fisher Information Metric (statistical, experimental) and Isometric Regularization (deterministic, foundational) converge. Patterns start as experimental, get promoted to foundational when evidence accumulates.

---

## Hardware Environment (CRITICAL - Read Before Starting)

**Your execution environment:**

| Resource | Spec | Constraint |
|----------|------|------------|
| CPU | AMD Ryzen 7 250, 8c/16t @ 5.1GHz | Strong |
| RAM | 38 GB free | Ample |
| Disk | 48 GB free | Tight - avoid large installs |
| GPU | None | CPU-only, no CUDA |

**Conda environments:**
- `tensor` (current): NumPy, SciPy, pandas, BeautifulSoup, SymPy — **Use for Layers 6, 8, 10**
- `dev-agent`: torch 2.10.0 CPU-only — **Use for Layer 9 only**

**Execution order:** 6 → 8 → 9 (switch env) → 10 (switch back)

**Critical rules:**
- Do NOT install causalnex (broken deps)
- Do NOT install large packages without checking `df -h` first
- Layer 9 requires `conda activate dev-agent` then switch back after
- All torch operations must use `torch.device('cpu')` explicitly

---

## Architecture Overview

### What Exists (144 tests passing)

```
unified-tensor-system/
├── tensor/
│   ├── core.py                    # ✅ 4-level tensor
│   ├── trajectory.py              # ✅ Lyapunov energy + WAL
│   ├── agent_network.py           # ✅ Thread-safe agents
│   ├── ficuts_updater.py          # ✅ Self-modifies this file
│   └── [Layers 6-10 to build below]
├── ecemath/                       # ✅ NumPy circuit math
├── dev-agent/                     # ✅ Intent-driven coder
├── run_system.py                  # ✅ Main runner
└── tests/                         # ✅ 144/144 passing
```

### What You'll Build (Layers 6-10)

**Layer 6:** Web ingestion (HTML parser, equation extractor, RSS feeds)  
**Layer 8:** Function basis (SymPy parser, library builder, HDV mapper)  
**Layer 9:** Dual geometry (FIM for experimental, IRMF for foundational)  
**Layer 10:** Multi-instance (spawn children, isometric transfer, aggregate)

---

## Dual Geometry Principle (Core Innovation)

The system operates on TWO manifolds:

### 1. Statistical Manifold (Fisher Information Metric)

**When to use:** Experimental patterns with uncertainty
- Market momentum (noisy, domain-specific)
- Empirical fits (might be wrong)
- Heuristics (high variance)

**Geometry:** g_ij = E[∂_i log p(x|θ) · ∂_j log p(x|θ)]

**Outputs:**
- Parameter importance (which θ_i matters most?)
- Uncertainty bounds (Cramér-Rao: var(θ) ≥ 1/FIM)
- Optimal sampling (where to gather next data?)

### 2. Deterministic Manifold (Isometric Regularization)

**When to use:** Foundational patterns that are exact
- Energy conservation (exact, not statistical)
- Wave equations (deterministic PDEs)
- Symmetry groups (provably correct)

**Geometry:** ||z₁ - z₂|| ≈ ||f(z₁) - f(z₂)|| (preserve distances)

**Outputs:**
- Smooth function basis manifold
- Geometric transfer between instances
- Robust interpolation

### The Bridge: Promotion

When experimental pattern shows:
- ✅ Low FIM uncertainty (< 0.01)
- ✅ Appears in ≥3 domains
- ✅ Passes conservation law test

→ **PROMOTE** from statistical → deterministic

This is how system learns what's genuinely universal vs domain-specific heuristic.

---

## Self-Modification Protocol

Five entities modify this file:

1. **Human:** Mark tasks in progress `[~]`, blocked `[⊗]`, add notes
2. **Claude Code (you):** Mark complete `[✓]`, update "Last Updated By"
3. **Claude Chat:** Clarify specs, add test cases
4. **Running System:** Log discoveries, update hypotheses
5. **Multi-Instance:** Report child discoveries

**How you update this file:**
```python
from tensor.ficuts_updater import FICUTSUpdater

ficuts = FICUTSUpdater('FICUTS.md')
ficuts.mark_task_complete('Task 6.1')
ficuts.update_field('Current Phase', 'Layer 8')
```

---

## Current Hypothesis

*System populates this as it discovers universals. Initially empty.*

**Hypothesis 1:** (awaiting first discovery)

---

## TASK LIST - Execute in Order

### Status Legend
- `[ ]` Not started
- `[~]` In progress  
- `[✓]` Complete, tests passing
- `[⊗]` Blocked (reason in notes)

---

### LAYER 1: Lyapunov Energy + WAL ✅ COMPLETE

All tasks done. 144/144 tests passing.

---

### LAYER 4: Concurrency + Memory ✅ COMPLETE

Thread-safe, memory-bounded. All tasks done.

---

### LAYER 5: FICUTS Self-Modification ✅ COMPLETE

FICUTSUpdater working. File self-updates.

---

### LAYER 6: Web Ingestion + Knowledge Extraction

**File:** `tensor/web_ingestion.py` (new)

**Purpose:** Ingest research papers, extract equations/parameters, build knowledge base

**Environment:** Stay in `tensor` (BeautifulSoup, requests, feedparser available)

---

#### Task 6.1: HTML Article Parser `[✓]`

**What:** Parse HTML → structured data (title, sections, links, code blocks, DOM depth)

**Code:**
```python
from bs4 import BeautifulSoup
from typing import Dict, List

class ArticleParser:
    """
    Extract structural information from HTML articles/papers.
    
    Captures:
    - DOM hierarchy (h1/h2/h3 structure)
    - Hyperlink graph (which concepts link where)
    - Code blocks (equations, algorithms)
    - Emphasis patterns (bold, italic, quotes)
    """
    
    def parse(self, html: str, url: str) -> Dict:
        soup = BeautifulSoup(html, 'html.parser')
        
        return {
            'url': url,
            'title': self._extract_title(soup),
            'sections': self._extract_sections(soup),
            'hyperlinks': self._extract_hyperlinks(soup),
            'code_blocks': self._extract_code(soup),
            'emphasis_map': self._extract_emphasis(soup),
            'dom_depth': self._compute_dom_depth(soup)
        }
    
    def _extract_title(self, soup) -> str:
        """Get title from <title> or <h1>"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.text.strip()
        h1 = soup.find('h1')
        return h1.text.strip() if h1 else 'Untitled'
    
    def _extract_sections(self, soup) -> List[Dict]:
        """Extract h1/h2/h3 hierarchy"""
        sections = []
        for tag in soup.find_all(['h1', 'h2', 'h3']):
            sections.append({
                'level': int(tag.name[1]),
                'text': tag.text.strip(),
                'position': len(sections)
            })
        return sections
    
    def _extract_hyperlinks(self, soup) -> List[Dict]:
        """Extract all <a> links with anchor text"""
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                'href': a['href'],
                'text': a.text.strip(),
                'outbound': not a['href'].startswith('#')
            })
        return links
    
    def _extract_code(self, soup) -> List[str]:
        """Extract <pre> code blocks and inline <code>"""
        code_blocks = [block.text.strip() for block in soup.find_all('pre')]
        inline_code = [code.text.strip() for code in soup.find_all('code')]
        return code_blocks + inline_code
    
    def _extract_emphasis(self, soup) -> Dict:
        """Count bold, italic, blockquote usage"""
        return {
            'bold_count': len(soup.find_all(['b', 'strong'])),
            'italic_count': len(soup.find_all(['i', 'em'])),
            'quote_count': len(soup.find_all('blockquote'))
        }
    
    def _compute_dom_depth(self, soup) -> int:
        """Maximum nesting depth of DOM tree"""
        def depth(elem):
            if not hasattr(elem, 'children'):
                return 0
            children_depths = [depth(c) for c in elem.children if hasattr(c, 'name')]
            return 1 + max(children_depths) if children_depths else 0
        return depth(soup)
```

**Test:**
```python
def test_article_parser():
    parser = ArticleParser()
    
    # Test with sample HTML
    html = """
    <html>
        <title>Test Article</title>
        <h1>Main Title</h1>
        <h2>Section 1</h2>
        <p>Content with <b>bold</b> and <a href="http://example.com">link</a></p>
        <pre>code block</pre>
    </html>
    """
    
    result = parser.parse(html, 'http://test.com')
    
    assert result['title'] == 'Test Article'
    assert len(result['sections']) == 2
    assert result['sections'][0]['level'] == 1
    assert len(result['hyperlinks']) > 0
    assert len(result['code_blocks']) > 0
    assert result['emphasis_map']['bold_count'] > 0
    assert result['dom_depth'] > 0
    
    print("[PASS] ArticleParser test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 6.2: Research Concept Extractor `[✓]`

**What:** Extract equations (LaTeX), parameters (Greek letters + values), technical terms

**Code:**
```python
import re
from typing import List, Dict

class ResearchConceptExtractor:
    """
    Extract mathematical/scientific concepts from parsed articles.
    
    Extracts:
    - Equations (LaTeX patterns: \\frac, \\int, \\partial, etc.)
    - Parameters (τ = 5ms, α = 0.01, etc.)
    - Technical terms (Title Case phrases)
    - Experimental indicators (procedure, measured, etc.)
    """
    
    def extract(self, article: Dict) -> Dict:
        return {
            'equations': self._extract_equations(article),
            'parameters': self._extract_parameters(article),
            'technical_terms': self._extract_technical_terms(article),
            'has_experiment': self._detect_experiment(article)
        }
    
    def _extract_equations(self, article: Dict) -> List[str]:
        """Find LaTeX or math patterns in code blocks"""
        equations = []
        latex_indicators = ['\\frac', '\\int', '\\sum', '\\partial', '\\Delta', '\\nabla']
        
        for code in article.get('code_blocks', []):
            if any(ind in code for ind in latex_indicators):
                # Clean up whitespace
                eq = ' '.join(code.split())
                equations.append(eq)
        
        return equations
    
    def _extract_parameters(self, article: Dict) -> List[Dict]:
        """
        Find parameter assignments like:
        - τ = 5ms
        - learning rate α = 0.01
        - sigma = 2.5
        """
        params = []
        
        # Greek letters pattern
        greek = r'[α-ωΑ-Ω]|alpha|beta|gamma|delta|epsilon|tau|phi|psi|omega|sigma|lambda|mu|nu|rho|kappa|theta|zeta|xi|eta|chi'
        
        # Get all text
        text = ' '.join([s['text'] for s in article.get('sections', [])])
        
        # Pattern: greek_letter = number [units]
        pattern = f'({greek})\\s*=\\s*([0-9.]+(?:e[+-]?[0-9]+)?)\\s*([a-zA-Z]*)'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            params.append({
                'symbol': match.group(1),
                'value': float(match.group(2)),
                'units': match.group(3) if match.group(3) else None,
                'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
        
        return params
    
    def _extract_technical_terms(self, article: Dict) -> List[str]:
        """Heuristic: Title Case multi-word phrases (2-4 words)"""
        text = ' '.join([s['text'] for s in article.get('sections', [])])
        
        # Pattern: Capitalized words (2-4 word phrases)
        pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b'
        terms = re.findall(pattern, text)
        
        # Remove duplicates, keep unique
        return list(set(terms))
    
    def _detect_experiment(self, article: Dict) -> bool:
        """Does article describe experimental procedure?"""
        indicators = [
            'we measured', 'experiment', 'experimental setup',
            'procedure', 'method', 'methodology', 
            'data collection', 'results', 'we observed'
        ]
        
        text = ' '.join([s['text'] for s in article.get('sections', [])]).lower()
        
        return any(ind in text for ind in indicators)
```

**Test:**
```python
def test_concept_extractor():
    extractor = ResearchConceptExtractor()
    
    article = {
        'sections': [
            {'text': 'We use exponential decay with tau = 5ms in our Neural Network model.'},
            {'text': 'The learning rate alpha = 0.01 was chosen empirically.'}
        ],
        'code_blocks': [
            '\\frac{dV}{dt} = -\\frac{V}{\\tau}',
            'y = A e^{-\\lambda t}'
        ]
    }
    
    result = extractor.extract(article)
    
    assert len(result['equations']) == 2
    assert any('frac' in eq for eq in result['equations'])
    
    assert len(result['parameters']) >= 2
    param_symbols = [p['symbol'] for p in result['parameters']]
    assert 'tau' in param_symbols or 'α' in param_symbols or 'alpha' in param_symbols
    
    assert len(result['technical_terms']) > 0
    assert result['has_experiment'] == False  # would be True if "we measured" present
    
    print("[PASS] ConceptExtractor test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 6.3: Web Ingestion Loop `[ ]`

**What:** Continuous scraping from RSS feeds (arXiv, blogs), deduplicate, store

**Code:**
```python
import feedparser
import requests
import time
from pathlib import Path
import json

class WebIngestionLoop:
    """
    Continuously ingest from web sources:
    - RSS feeds (arXiv cs.AI, cs.LG, physics, etc.)
    - News aggregators
    - Research blogs
    
    Deduplicate by URL, parse, extract concepts, store.
    """
    
    def __init__(self, storage_dir='tensor/data/ingested'):
        self.parser = ArticleParser()
        self.extractor = ResearchConceptExtractor()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load seen URLs
        self.seen_file = self.storage_dir / 'seen_urls.json'
        if self.seen_file.exists():
            self.seen_urls = set(json.loads(self.seen_file.read_text()))
        else:
            self.seen_urls = set()
    
    def ingest_url(self, url: str) -> bool:
        """
        Fetch single URL, parse, extract, store.
        
        Returns: True if successfully ingested, False if duplicate or error
        """
        if url in self.seen_urls:
            return False  # duplicate
        
        try:
            # Fetch
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse
            article = self.parser.parse(response.text, url)
            
            # Extract concepts
            concepts = self.extractor.extract(article)
            
            # Store
            self._store_ingested(url, article, concepts)
            
            # Mark as seen
            self.seen_urls.add(url)
            self._save_seen_urls()
            
            print(f"[WebIngestion] Ingested: {url}")
            return True
            
        except Exception as e:
            print(f"[WebIngestion] Failed {url}: {e}")
            return False
    
    def run_continuous(self, feed_urls: List[str], interval_seconds=3600):
        """
        Continuously ingest from RSS feeds.
        
        feed_urls: List of RSS feed URLs (e.g., arXiv RSS)
        interval_seconds: How often to check feeds (default 1 hour)
        """
        print(f"[WebIngestion] Starting continuous loop with {len(feed_urls)} feeds")
        
        while True:
            for feed_url in feed_urls:
                articles = self._fetch_feed(feed_url)
                print(f"[WebIngestion] Found {len(articles)} articles in {feed_url}")
                
                for article_url in articles:
                    self.ingest_url(article_url)
            
            print(f"[WebIngestion] Sleeping {interval_seconds}s until next check")
            time.sleep(interval_seconds)
    
    def _fetch_feed(self, feed_url: str) -> List[str]:
        """Parse RSS/Atom feed, return article URLs"""
        try:
            feed = feedparser.parse(feed_url)
            return [entry.link for entry in feed.entries]
        except Exception as e:
            print(f"[WebIngestion] Feed parse error {feed_url}: {e}")
            return []
    
    def _store_ingested(self, url: str, article: Dict, concepts: Dict):
        """Store parsed article + extracted concepts as JSON"""
        # Create unique filename from URL
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        filename = self.storage_dir / f"{url_hash}.json"
        
        data = {
            'url': url,
            'article': article,
            'concepts': concepts,
            'ingested_at': time.time()
        }
        
        filename.write_text(json.dumps(data, indent=2))
    
    def _save_seen_urls(self):
        """Persist seen URLs to disk"""
        self.seen_file.write_text(json.dumps(list(self.seen_urls)))
    
    def get_ingested_count(self) -> int:
        """Count how many articles stored"""
        return len(list(self.storage_dir.glob('*.json')))
```

**Test:**
```python
def test_web_ingestion():
    loop = WebIngestionLoop(storage_dir='tensor/data/test_ingest')
    
    # Use arXiv cs.AI RSS feed
    feed_urls = ['http://export.arxiv.org/rss/cs.AI']
    
    # Ingest just first 5 articles
    articles = loop._fetch_feed(feed_urls[0])
    assert len(articles) > 0, "No articles in feed"
    
    for url in articles[:5]:
        loop.ingest_url(url)
    
    # Check storage
    assert loop.get_ingested_count() >= 5
    
    # Test deduplication
    initial_count = loop.get_ingested_count()
    loop.ingest_url(articles[0])  # try to ingest again
    assert loop.get_ingested_count() == initial_count  # should not increase
    
    print(f"[PASS] WebIngestion test - ingested {loop.get_ingested_count()} articles")
```

**Status:** `[ ]`  
**Notes:**

---

### LAYER 8: Universal Function Basis (from Papers)

**File:** `tensor/function_basis.py` (new)

**Purpose:** Parse equations → classify function types → build library → map to HDV space

**Environment:** Stay in `tensor` (SymPy available)

---

#### Task 8.1: Symbolic Equation Parser `[ ]`

**What:** LaTeX → SymPy, classify function type, extract parameters

**Code:**
```python
import sympy as sp
from sympy.parsing.latex import parse_latex
from typing import List, Optional

class EquationParser:
    """
    Parse LaTeX equations → SymPy symbolic expressions.
    Classify function type, extract parameters.
    """
    
    def parse(self, latex_string: str) -> Optional[sp.Expr]:
        """
        Convert LaTeX → SymPy expression.
        
        Examples:
        "\\frac{dV}{dt} = -\\frac{V}{\\tau}" → sp.Derivative(V, t) == -V/tau
        "y = A e^{-\\lambda t}" → y == A*exp(-lambda*t)
        """
        try:
            # Clean up common LaTeX issues
            cleaned = latex_string.replace('\\,', ' ').replace('\\;', ' ')
            expr = parse_latex(cleaned)
            return expr
        except Exception as e:
            # Fallback: try simple pattern matching
            return self._parse_simple(latex_string)
    
    def _parse_simple(self, latex_string: str) -> Optional[sp.Expr]:
        """Fallback parser for common patterns"""
        # Exponential: e^{-t/tau}
        if 'e^{' in latex_string:
            # Extract exponent
            import re
            match = re.search(r'e\^\{([^}]+)\}', latex_string)
            if match:
                exponent_str = match.group(1)
                # Try to parse exponent
                try:
                    exponent = sp.sympify(exponent_str)
                    return sp.exp(exponent)
                except:
                    pass
        
        return None
    
    def classify_function_type(self, expr: sp.Expr) -> str:
        """
        Identify function type from expression.
        
        Types:
        - exponential: contains exp()
        - power_law: x^α where α is parameter
        - polynomial: polynomial in x
        - trigonometric: sin/cos/tan
        - logarithmic: log/ln
        - special: Bessel, hypergeometric, etc.
        """
        if expr is None:
            return 'unknown'
        
        expr_str = str(expr)
        
        # Check for function types
        if 'exp' in expr_str:
            return 'exponential'
        elif any(fn in expr_str for fn in ['sin', 'cos', 'tan', 'Sin', 'Cos', 'Tan']):
            return 'trigonometric'
        elif any(fn in expr_str for fn in ['log', 'ln', 'Log']):
            return 'logarithmic'
        elif 'Pow' in expr_str or '**' in expr_str:
            # Check if exponent is constant (polynomial) or variable (power law)
            # Heuristic: if exponent contains t or x, it's power law
            if 't' in expr_str or 'x' in expr_str:
                return 'power_law'
            return 'polynomial'
        else:
            return 'algebraic'
    
    def extract_parameters(self, expr: sp.Expr) -> List[str]:
        """
        Extract all parameters (symbols that aren't independent variables).
        
        Heuristic: t, x, y, z are independent variables
        Everything else is a parameter
        """
        if expr is None:
            return []
        
        # Independent variables
        independent = {'t', 'x', 'y', 'z', 'T', 'X', 'Y', 'Z'}
        
        # Get all symbols
        all_symbols = expr.free_symbols
        
        # Filter to just parameters
        params = [str(sym) for sym in all_symbols if str(sym) not in independent]
        
        return sorted(params)
```

**Test:**
```python
def test_equation_parser():
    parser = EquationParser()
    
    # Test exponential
    expr1 = parser.parse("e^{-t/\\tau}")
    assert expr1 is not None
    assert parser.classify_function_type(expr1) == 'exponential'
    params1 = parser.extract_parameters(expr1)
    assert 'tau' in params1 or 'τ' in params1
    
    # Test trigonometric
    expr2 = sp.sin(sp.Symbol('omega') * sp.Symbol('t'))
    assert parser.classify_function_type(expr2) == 'trigonometric'
    
    print("[PASS] EquationParser test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 8.2: Function Basis Library Builder `[ ]`

**What:** Aggregate equations from all ingested papers, deduplicate, track domains

**Code:**
```python
import json
from pathlib import Path
from collections import defaultdict
import time

class FunctionBasisLibrary:
    """
    Aggregated function library discovered from papers.
    
    Structure:
    {
      'exponential_decay_0': {
        'symbolic': sp.exp(-t/tau),
        'type': 'exponential',
        'parameters': ['tau'],
        'domains': {'ece', 'biology', 'finance'},
        'source_papers': ['paper_id_1', 'paper_id_2'],
        'classification': 'experimental',  # will become 'foundational' if universal
        'discovered_at': timestamp
      },
      ...
    }
    """
    
    def __init__(self, library_path='tensor/data/function_library.json'):
        self.library_path = Path(library_path)
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing library
        if self.library_path.exists():
            self.library = json.loads(self.library_path.read_text())
            # Convert string representations back to SymPy (if needed for computation)
        else:
            self.library = {}
        
        self.parser = EquationParser()
    
    def ingest_papers_from_storage(self, storage_dir='tensor/data/ingested'):
        """
        Load all ingested papers, extract equations, add to library.
        """
        storage = Path(storage_dir)
        paper_files = list(storage.glob('*.json'))
        
        print(f"[FunctionLibrary] Processing {len(paper_files)} papers")
        
        for paper_file in paper_files:
            paper_data = json.loads(paper_file.read_text())
            
            paper_id = paper_file.stem
            url = paper_data['url']
            
            # Infer domain from URL or title
            domain = self._infer_domain(url, paper_data['article'].get('title', ''))
            
            # Extract equations
            equations = paper_data['concepts'].get('equations', [])
            
            for eq_latex in equations:
                self._add_equation(paper_id, eq_latex, domain)
        
        self._save_library()
        print(f"[FunctionLibrary] Library now has {len(self.library)} functions")
    
    def _add_equation(self, paper_id: str, latex: str, domain: str):
        """Parse equation and add to library"""
        expr = self.parser.parse(latex)
        if expr is None:
            return  # failed to parse
        
        func_type = self.parser.classify_function_type(expr)
        params = self.parser.extract_parameters(expr)
        
        # Check if this function already exists (symbolic equivalence)
        existing_name = self._find_matching_function(expr)
        
        if existing_name:
            # Update existing entry
            self.library[existing_name]['domains'].add(domain)
            self.library[existing_name]['source_papers'].append(paper_id)
        else:
            # Add new function
            func_name = f"{func_type}_{len(self.library)}"
            
            self.library[func_name] = {
                'symbolic_str': str(expr),  # store as string for JSON
                'type': func_type,
                'parameters': params,
                'domains': {domain},  # will be converted to list for JSON
                'source_papers': [paper_id],
                'classification': 'experimental',  # starts as experimental
                'discovered_at': time.time()
            }
    
    def _find_matching_function(self, expr: sp.Expr) -> Optional[str]:
        """Check if expression matches existing function"""
        for name, data in self.library.items():
            try:
                existing_expr = sp.sympify(data['symbolic_str'])
                if sp.simplify(expr - existing_expr) == 0:
                    return name
            except:
                continue
        return None
    
    def _infer_domain(self, url: str, title: str) -> str:
        """Heuristic: infer domain from URL or title"""
        text = (url + ' ' + title).lower()
        
        if any(kw in text for kw in ['circuit', 'electronic', 'vlsi', 'semiconductor']):
            return 'ece'
        elif any(kw in text for kw in ['biology', 'neuron', 'synapse', 'cell']):
            return 'biology'
        elif any(kw in text for kw in ['finance', 'market', 'trading', 'economics']):
            return 'finance'
        elif any(kw in text for kw in ['physics', 'mechanics', 'quantum']):
            return 'physics'
        else:
            return 'general'
    
    def get_universal_functions(self, min_domains=3) -> List[str]:
        """Return functions appearing in ≥N domains (likely universal)"""
        return [
            name for name, data in self.library.items()
            if len(data['domains']) >= min_domains
        ]
    
    def promote_to_foundational(self, func_name: str):
        """Mark function as foundational (provably universal)"""
        if func_name in self.library:
            self.library[func_name]['classification'] = 'foundational'
            self._save_library()
    
    def _save_library(self):
        """Persist library to disk (convert sets to lists for JSON)"""
        serializable = {}
        for name, data in self.library.items():
            serializable[name] = data.copy()
            if isinstance(data['domains'], set):
                serializable[name]['domains'] = list(data['domains'])
        
        self.library_path.write_text(json.dumps(serializable, indent=2))
```

**Test:**
```python
def test_function_library():
    # First run web ingestion to get some papers
    # (Assuming Task 6.3 has run and ingested some papers)
    
    library = FunctionBasisLibrary(library_path='tensor/data/test_library.json')
    library.ingest_papers_from_storage('tensor/data/test_ingest')
    
    assert len(library.library) > 0, "No functions in library"
    
    # Check if any universal functions detected
    universals = library.get_universal_functions(min_domains=2)
    print(f"[FunctionLibrary] Found {len(universals)} functions in ≥2 domains")
    
    # If exponential decay found across domains, it should be there
    exp_funcs = [name for name in library.library if 'exponential' in name]
    if exp_funcs:
        print(f"[FunctionLibrary] Exponential functions: {exp_funcs}")
    
    print("[PASS] FunctionLibrary test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 8.3: Function Basis → HDV Dimension Mapping `[ ]`

**What:** Assign each basis function to specific HDV dimensions

**Code:**
```python
import numpy as np

class FunctionBasisToHDV:
    """
    Map function library → HDV space dimensions.
    
    Strategy:
    - Foundational functions → low dimensions (0-99)
    - Universal functions (≥3 domains) → multiple dimensions (redundancy)
    - Experimental functions → higher dimensions
    
    Result: Each domain gets a mask showing which HDV dims it uses
    """
    
    def __init__(self, function_library: FunctionBasisLibrary, hdv_dim=10000):
        self.library = function_library
        self.hdv_dim = hdv_dim
        self.dim_assignments = {}  # func_name → dim_id or [dim_ids]
        self.next_free_dim = 0
    
    def assign_dimensions(self):
        """
        Assign HDV dimensions to all functions in library.
        
        Priority order:
        1. Foundational → dims 0-99
        2. Universal (≥3 domains) → multiple dims for robustness
        3. Experimental → single dims
        """
        # Sort functions by priority
        funcs = list(self.library.library.items())
        
        def priority(item):
            name, data = item
            is_foundational = data['classification'] == 'foundational'
            domain_count = len(data['domains'])
            return (
                0 if is_foundational else 1,  # foundational first
                -domain_count  # then by domain count (descending)
            )
        
        funcs.sort(key=priority)
        
        for func_name, func_data in funcs:
            if func_data['classification'] == 'foundational':
                # Assign single low dimension
                self.dim_assignments[func_name] = self.next_free_dim
                self.next_free_dim += 1
                
            elif len(func_data['domains']) >= 3:
                # Universal: assign multiple dimensions (5 for redundancy)
                dims = list(range(self.next_free_dim, self.next_free_dim + 5))
                self.dim_assignments[func_name] = dims
                self.next_free_dim += 5
                
            else:
                # Experimental: single dimension
                self.dim_assignments[func_name] = self.next_free_dim
                self.next_free_dim += 1
        
        print(f"[HDVMapping] Assigned {self.next_free_dim} dimensions to {len(funcs)} functions")
    
    def get_domain_mask(self, domain: str) -> np.ndarray:
        """
        Generate binary mask for a domain.
        
        mask[i] = 1 if domain uses the function assigned to dimension i
        """
        mask = np.zeros(self.hdv_dim, dtype=bool)
        
        for func_name, func_data in self.library.library.items():
            if domain in func_data.get('domains', set()):
                # This domain uses this function
                dims = self.dim_assignments.get(func_name, [])
                
                # Handle both single int and list of ints
                if isinstance(dims, int):
                    dims = [dims]
                
                for d in dims:
                    if d < self.hdv_dim:
                        mask[d] = True
        
        return mask
    
    def get_overlap_dimensions(self) -> set:
        """
        Find dimensions used by ≥2 domains (overlaps = where universals live).
        """
        domain_masks = {}
        
        # Get unique domains
        all_domains = set()
        for func_data in self.library.library.values():
            all_domains.update(func_data.get('domains', set()))
        
        # Get mask for each domain
        for domain in all_domains:
            domain_masks[domain] = self.get_domain_mask(domain)
        
        # Count usage per dimension
        usage_count = np.zeros(self.hdv_dim, dtype=int)
        for mask in domain_masks.values():
            usage_count += mask.astype(int)
        
        # Overlaps = used by ≥2 domains
        overlap_dims = set(np.where(usage_count >= 2)[0].tolist())
        
        return overlap_dims
```

**Test:**
```python
def test_hdv_mapping():
    library = FunctionBasisLibrary(library_path='tensor/data/test_library.json')
    library.ingest_papers_from_storage('tensor/data/test_ingest')
    
    mapper = FunctionBasisToHDV(library, hdv_dim=10000)
    mapper.assign_dimensions()
    
    # Get mask for ECE domain
    ece_mask = mapper.get_domain_mask('ece')
    assert ece_mask.sum() > 0, "ECE mask should have some active dimensions"
    
    # Get overlaps
    overlaps = mapper.get_overlap_dimensions()
    print(f"[HDVMapping] Found {len(overlaps)} overlap dimensions")
    
    # If we have functions in ≥2 domains, should have overlaps
    universals = library.get_universal_functions(min_domains=2)
    if universals:
        assert len(overlaps) > 0, "Should have overlaps if universal functions exist"
    
    print("[PASS] HDVMapping test")
```

**Status:** `[ ]`  
**Notes:**

---

### LAYER 9: Dual Geometry (FIM + IRMF)

**Files:** `tensor/dual_geometry.py`, `tensor/fisher_manifold.py`, `tensor/isometric_manifold.py` (new)

**Purpose:** Learn patterns on two manifolds (statistical + deterministic), promote when criteria met

**Environment:** **SWITCH TO dev-agent** (`conda activate dev-agent`) — torch required

**CRITICAL:** After Layer 9, switch back to `tensor` env

---

#### Task 9.1: Fisher Information Manifold `[ ]`

**What:** Learn experimental patterns, compute FIM, track uncertainty

**Code:**
```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple

class FisherInformationManifold:
    """
    Statistical manifold for experimental patterns.
    
    Learns p(x | θ) from data.
    Computes Fisher Information Matrix: g_ij = E[∂_i log p · ∂_j log p]
    Provides uncertainty bounds via Cramér-Rao: var(θ_i) ≥ 1/g_ii
    """
    
    def __init__(self):
        self.patterns = {}  # pattern_id → {theta, FIM, uncertainty}
    
    def learn_distribution(self, pattern_id: str, data: np.ndarray, 
                          initial_theta: np.ndarray) -> Dict:
        """
        Fit p(x | θ) to data via MLE.
        Compute FIM at MLE point.
        
        Returns: {theta_mle, FIM, uncertainty_bounds}
        """
        # Maximum Likelihood Estimation
        def neg_log_likelihood(theta):
            return -self._log_likelihood(data, theta)
        
        result = minimize(neg_log_likelihood, initial_theta, method='BFGS')
        theta_mle = result.x
        
        # Compute Fisher Information Matrix
        fim = self._compute_fim(theta_mle, data)
        
        # Cramér-Rao bound: var(theta_i) ≥ 1 / FIM_ii
        try:
            fim_inv = np.linalg.inv(fim)
            uncertainty = np.sqrt(np.diag(fim_inv))
        except np.linalg.LinAlgError:
            # FIM is singular
            uncertainty = np.inf * np.ones(len(theta_mle))
        
        # Store
        self.patterns[pattern_id] = {
            'theta': theta_mle,
            'FIM': fim,
            'uncertainty': uncertainty,
            'data_size': len(data)
        }
        
        return self.patterns[pattern_id]
    
    def _log_likelihood(self, data: np.ndarray, theta: np.ndarray) -> float:
        """
        Log-likelihood for Gaussian model: p(x | μ, σ) = N(μ, σ²)
        
        For other distributions, override this method.
        """
        if len(theta) != 2:
            # For simplicity, assume Gaussian with μ, σ
            theta = np.array([np.mean(data), np.std(data)])
        
        mu, sigma = theta[0], max(theta[1], 1e-6)  # prevent sigma=0
        
        log_p = -0.5 * np.log(2 * np.pi * sigma**2) - (data - mu)**2 / (2 * sigma**2)
        return np.sum(log_p)
    
    def _compute_fim(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Fisher Information Matrix via numerical differentiation.
        
        FIM_ij = -E[∂²log p / ∂θ_i ∂θ_j]
        
        For Gaussian: FIM = [[n/σ², 0], [0, 2n/σ²]]
        """
        n = len(data)
        d = len(theta)
        fim = np.zeros((d, d))
        
        epsilon = 1e-5
        
        for i in range(d):
            for j in range(d):
                # Compute second derivative via finite differences
                theta_pp = theta.copy(); theta_pp[i] += epsilon; theta_pp[j] += epsilon
                theta_pm = theta.copy(); theta_pm[i] += epsilon; theta_pm[j] -= epsilon
                theta_mp = theta.copy(); theta_mp[i] -= epsilon; theta_mp[j] += epsilon
                theta_mm = theta.copy(); theta_mm[i] -= epsilon; theta_mm[j] -= epsilon
                
                d2L = (
                    self._log_likelihood(data, theta_pp)
                    - self._log_likelihood(data, theta_pm)
                    - self._log_likelihood(data, theta_mp)
                    + self._log_likelihood(data, theta_mm)
                ) / (4 * epsilon**2)
                
                fim[i, j] = -d2L
        
        return fim
    
    def is_ready_for_promotion(self, pattern_id: str, threshold=0.01) -> bool:
        """
        Check if pattern can be promoted to foundational.
        
        Criteria: All parameter uncertainties < threshold
        """
        if pattern_id not in self.patterns:
            return False
        
        uncertainty = self.patterns[pattern_id]['uncertainty']
        return np.all(uncertainty < threshold)
    
    def get_most_informative_parameters(self, pattern_id: str) -> np.ndarray:
        """
        Return parameter indices sorted by information (descending).
        
        High FIM_ii = low uncertainty = more informative
        """
        fim = self.patterns[pattern_id]['FIM']
        info = np.diag(fim)
        return np.argsort(info)[::-1]  # descending order
```

**Test:**
```python
def test_fisher_manifold():
    fim = FisherInformationManifold()
    
    # Generate synthetic Gaussian data
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, size=1000)
    
    # Fit
    initial = np.array([0.0, 1.0])
    result = fim.learn_distribution('test_pattern', data, initial)
    
    # Check MLE close to true values
    assert abs(result['theta'][0] - true_mu) < 0.1
    assert abs(result['theta'][1] - true_sigma) < 0.2
    
    # Check uncertainty bounds
    assert result['uncertainty'][0] < 0.1  # should be small for large n
    
    # Check promotion readiness
    ready = fim.is_ready_for_promotion('test_pattern', threshold=0.1)
    assert ready  # should be ready with n=1000
    
    print("[PASS] FisherInformationManifold test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 9.2: Isometric Regularization Manifold `[ ]`

**What:** Learn deterministic function manifold with distance preservation

**Code:**
```python
import torch
import torch.nn as nn
from typing import List, Tuple

class IsometricFunctionManifold:
    """
    Deterministic function manifold for foundational patterns.
    
    Maps latent z → function f(x; z) such that:
    ||z1 - z2|| ≈ ||f(z1) - f(z2)||  (isometric: preserve distances)
    
    Based on IRMF paper principles.
    """
    
    def __init__(self, latent_dim=128, hidden_dim=256):
        self.latent_dim = latent_dim
        self.device = torch.device('cpu')  # CPU-only
        
        # Decoder: (z, x) → f(x)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        self.patterns = {}  # pattern_id → latent_vector z
    
    def learn_function(self, pattern_id: str, 
                      data_points: List[Tuple[float, float]],
                      n_epochs=1000) -> torch.Tensor:
        """
        Learn latent representation z for function defined by data_points.
        
        data_points: [(x1, y1), (x2, y2), ...]
        
        Returns: learned latent vector z
        """
        # Convert to tensors
        x = torch.tensor([p[0] for p in data_points], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor([p[1] for p in data_points], dtype=torch.float32).unsqueeze(1)
        
        # Initialize latent vector
        z = torch.randn(self.latent_dim, requires_grad=True, device=self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam([z], lr=0.01)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            z_expanded = z.unsqueeze(0).expand(x.shape[0], -1)
            inputs = torch.cat([z_expanded, x], dim=1)
            y_pred = self.decoder(inputs)
            
            # Reconstruction loss
            recon_loss = torch.mean((y_pred - y)**2)
            
            # Backward
            recon_loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"[IRMF] Epoch {epoch}, Loss: {recon_loss.item():.6f}")
        
        # Store learned latent
        self.patterns[pattern_id] = z.detach().clone()
        
        return z.detach()
    
    def generate_function_values(self, pattern_id: str, x_values: np.ndarray) -> np.ndarray:
        """
        Generate function f(x) values using learned latent z.
        """
        z = self.patterns[pattern_id]
        x = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)
        
        z_expanded = z.unsqueeze(0).expand(x.shape[0], -1)
        inputs = torch.cat([z_expanded, x], dim=1)
        
        with torch.no_grad():
            y = self.decoder(inputs)
        
        return y.numpy().flatten()
    
    def compute_isometric_loss(self, latent_vectors: torch.Tensor, n_pairs=100) -> torch.Tensor:
        """
        Isometric regularization loss (from IRMF paper).
        
        Ensure: ||z1 - z2|| ≈ ||f(z1) - f(z2)||
        
        This prevents distortion when learning function manifold.
        """
        # Sample pairs
        n = latent_vectors.shape[0]
        if n < 2:
            return torch.tensor(0.0)
        
        indices = torch.randint(0, n, (n_pairs, 2))
        z1 = latent_vectors[indices[:, 0]]
        z2 = latent_vectors[indices[:, 1]]
        
        # Distance in latent space
        d_latent = torch.norm(z1 - z2, dim=1)
        
        # Distance in function space (sample domain)
        x_samples = torch.linspace(0, 1, 50).unsqueeze(1)
        
        d_function = torch.zeros(n_pairs)
        for i in range(n_pairs):
            z1_expanded = z1[i].unsqueeze(0).expand(x_samples.shape[0], -1)
            z2_expanded = z2[i].unsqueeze(0).expand(x_samples.shape[0], -1)
            
            inputs1 = torch.cat([z1_expanded, x_samples], dim=1)
            inputs2 = torch.cat([z2_expanded, x_samples], dim=1)
            
            with torch.no_grad():
                f1 = self.decoder(inputs1)
                f2 = self.decoder(inputs2)
            
            d_function[i] = torch.norm(f1 - f2)
        
        # Penalize discrepancy
        loss = torch.mean((d_latent - d_function)**2)
        return loss
```

**Test:**
```python
def test_isometric_manifold():
    irmf = IsometricFunctionManifold(latent_dim=64, hidden_dim=128)
    
    # Generate data from exponential decay
    x = np.linspace(0, 5, 50)
    y = np.exp(-x / 2.0)  # tau=2
    data = [(x[i], y[i]) for i in range(len(x))]
    
    # Learn
    z = irmf.learn_function('exp_decay', data, n_epochs=500)
    
    assert z is not None
    assert z.shape == (64,)
    
    # Generate and check
    x_test = np.linspace(0, 5, 20)
    y_pred = irmf.generate_function_values('exp_decay', x_test)
    y_true = np.exp(-x_test / 2.0)
    
    error = np.mean((y_pred - y_true)**2)
    assert error < 0.1, f"Reconstruction error too high: {error}"
    
    print("[PASS] IsometricFunctionManifold test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 9.3: Dual Geometry Integration `[ ]`

**What:** Route patterns to appropriate manifold, promote when ready

**Code:**
```python
class DualGeometrySystem:
    """
    Manages both manifolds, routes patterns appropriately.
    
    Experimental patterns (noisy, uncertain) → FIM
    Foundational patterns (exact, universal) → IRMF
    
    Promotes experimental → foundational when:
    - FIM uncertainty < 0.01
    - Pattern in ≥3 domains
    - Passes conservation law test (TODO)
    """
    
    def __init__(self):
        self.fisher = FisherInformationManifold()
        self.isometric = IsometricFunctionManifold()
        
    def classify_pattern(self, pattern: Dict) -> str:
        """
        Decide: statistical or deterministic?
        
        Statistical signals:
        - High variance in observations
        - Domain-specific (< 3 domains)
        - Empirically fitted
        
        Deterministic signals:
        - Low variance
        - Cross-domain (≥3 domains)
        - Conservation law present
        """
        variance = np.var(pattern.get('observations', []))
        domain_count = len(pattern.get('domains', []))
        has_conservation = pattern.get('conserved_quantity', False)
        
        if variance > 0.1 or domain_count < 3 or not has_conservation:
            return 'statistical'
        else:
            return 'deterministic'
    
    def learn_pattern(self, pattern_id: str, pattern: Dict, data: np.ndarray):
        """Route to appropriate manifold"""
        classification = self.classify_pattern(pattern)
        
        if classification == 'statistical':
            print(f"[DualGeometry] Learning {pattern_id} on FIM (experimental)")
            initial_theta = np.array([np.mean(data), np.std(data)])
            self.fisher.learn_distribution(pattern_id, data, initial_theta)
            
        else:
            print(f"[DualGeometry] Learning {pattern_id} on IRMF (foundational)")
            # Convert data to (x, y) pairs
            x = np.linspace(0, 1, len(data))
            data_points = [(x[i], data[i]) for i in range(len(data))]
            self.isometric.learn_function(pattern_id, data_points)
    
    def promote_pattern(self, pattern_id: str, function_library: FunctionBasisLibrary) -> bool:
        """
        Attempt to promote experimental → foundational.
        
        Returns: True if promoted, False otherwise
        """
        # Check if in FIM
        if pattern_id not in self.fisher.patterns:
            return False
        
        # Check criteria
        if not self.fisher.is_ready_for_promotion(pattern_id, threshold=0.01):
            print(f"[DualGeometry] {pattern_id} not ready - uncertainty too high")
            return False
        
        # Check domain count (need function library to know)
        # This is simplified - in reality would check actual domains
        
        print(f"[DualGeometry] PROMOTING {pattern_id}: FIM → IRMF")
        
        # Mark in function library
        function_library.promote_to_foundational(pattern_id)
        
        # TODO: Transfer learned distribution to IRMF manifold
        # For now, just mark as promoted
        
        return True
```

**Test:**
```python
def test_dual_geometry():
    dual = DualGeometrySystem()
    
    # Test statistical pattern (high variance)
    pattern_exp = {
        'observations': np.random.randn(100) + 5,  # noisy
        'domains': ['ece'],  # only 1 domain
        'conserved_quantity': False
    }
    
    assert dual.classify_pattern(pattern_exp) == 'statistical'
    
    # Test deterministic pattern (low variance, multi-domain)
    pattern_found = {
        'observations': np.ones(100) * 5 + np.random.randn(100) * 0.01,  # low noise
        'domains': ['ece', 'biology', 'finance'],  # 3 domains
        'conserved_quantity': True
    }
    
    assert dual.classify_pattern(pattern_found) == 'deterministic'
    
    print("[PASS] DualGeometrySystem test")
```

**Status:** `[ ]`  
**Notes:**

**CRITICAL:** After completing Layer 9, switch back to `tensor` environment:
```bash
conda deactivate
conda activate tensor
```

---

### LAYER 10: Multi-Instance Coordination

**File:** `tensor/multi_instance.py` (new)

**Purpose:** Spawn child instances to explore HDV in parallel, communicate via isometric transfer

**Environment:** **Back to `tensor`** (pure Python multiprocessing)

---

#### Task 10.1: Instance Spawning and Management `[ ]`

**What:** Parent spawns children to explore different HDV regions in parallel

**Code:**
```python
import multiprocessing as mp
import numpy as np
from typing import List, Dict
import time

class MultiInstanceCoordinator:
    """
    Parent instance manages multiple child instances.
    Each child explores a different region of HDV space (a "chart").
    """
    
    def __init__(self, max_instances=4):
        self.max_instances = max_instances
        self.instances = []
        self.result_queue = mp.Queue()
        self.chart_centers = {}  # instance_id → HDV center point
        
    def should_spawn_child(self, ignorance_map: np.ndarray, 
                          learning_priority: np.ndarray) -> bool:
        """
        Decide when to spawn based on:
        - High ignorance (prediction error high in some region)
        - High learning priority (FIM says region is informative)
        - Not already covered by existing instances
        """
        if len(self.instances) >= self.max_instances:
            return False
        
        # Find region with highest ignorance × priority
        combined = ignorance_map * learning_priority
        max_idx = np.argmax(combined)
        
        # Check if any existing instance covers this region
        for instance_id, center in self.chart_centers.items():
            dist = np.linalg.norm(combined - center)
            if dist < 0.5:  # threshold
                return False  # already covered
        
        return combined[max_idx] > 0.3  # spawn threshold
    
    def spawn_child(self, chart_center: np.ndarray, exploration_radius=0.5) -> int:
        """
        Spawn child instance.
        
        chart_center: Point in HDV space to focus on
        exploration_radius: How far from center to explore
        """
        instance_id = len(self.instances)
        
        child = InstanceWorker(
            instance_id=instance_id,
            chart_center=chart_center,
            radius=exploration_radius,
            result_queue=self.result_queue
        )
        
        child.start()
        self.instances.append(child)
        self.chart_centers[instance_id] = chart_center
        
        print(f"[MultiInstance] Spawned child {instance_id} at center {chart_center[:3]}...")
        return instance_id
    
    def collect_results(self, timeout=1.0) -> List[Dict]:
        """
        Collect discoveries from all children (non-blocking).
        """
        results = []
        
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except:
                break
        
        return results
    
    def shutdown_all(self):
        """Terminate all child instances"""
        for instance in self.instances:
            instance.terminate()
            instance.join()
        
        self.instances = []
        self.chart_centers = {}


class InstanceWorker(mp.Process):
    """
    Child instance worker process.
    Explores assigned HDV region, reports discoveries to parent.
    """
    
    def __init__(self, instance_id: int, chart_center: np.ndarray, 
                 radius: float, result_queue: mp.Queue):
        super().__init__()
        self.instance_id = instance_id
        self.chart_center = chart_center
        self.radius = radius
        self.result_queue = result_queue
        
    def run(self):
        """
        Main exploration loop.
        """
        print(f"[Instance {self.instance_id}] Starting exploration")
        
        # Sample local neighborhood
        samples = self._sample_neighborhood(n_samples=100)
        
        # Evaluate each
        for i, sample in enumerate(samples):
            value = self._evaluate_sample(sample)
            
            if value > 0.7:  # discovery threshold
                self._report_discovery(sample, value)
        
        # Report completion
        self.result_queue.put({
            'instance_id': self.instance_id,
            'type': 'complete',
            'samples_evaluated': len(samples)
        })
        
        print(f"[Instance {self.instance_id}] Completed")
    
    def _sample_neighborhood(self, n_samples=100) -> List[np.ndarray]:
        """Sample points within radius of chart_center"""
        samples = []
        for _ in range(n_samples):
            offset = np.random.randn(len(self.chart_center)) * self.radius
            samples.append(self.chart_center + offset)
        return samples
    
    def _evaluate_sample(self, point: np.ndarray) -> float:
        """
        Evaluate quality of HDV point.
        
        In full system: would test if this point represents a good
        universal function candidate.
        
        Placeholder: random score
        """
        # Placeholder evaluation
        return np.random.rand()
    
    def _report_discovery(self, point: np.ndarray, value: float):
        """Send discovery to parent via queue"""
        self.result_queue.put({
            'instance_id': self.instance_id,
            'type': 'discovery',
            'point': point.tolist(),  # convert to list for JSON
            'value': value,
            'timestamp': time.time()
        })
```

**Test:**
```python
def test_multi_instance():
    coordinator = MultiInstanceCoordinator(max_instances=3)
    
    # Spawn 3 children with different centers
    centers = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    
    for center in centers:
        coordinator.spawn_child(center, exploration_radius=0.3)
    
    # Wait a bit for them to work
    time.sleep(2)
    
    # Collect results
    results = coordinator.collect_results(timeout=0.5)
    
    assert len(results) > 0, "Should have some results"
    
    # Check for discoveries
    discoveries = [r for r in results if r['type'] == 'discovery']
    print(f"[MultiInstance] Collected {len(discoveries)} discoveries")
    
    # Shutdown
    coordinator.shutdown_all()
    
    print("[PASS] MultiInstance test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 10.2: Isometric Transfer Between Instances `[ ]`

**What:** Transfer discoveries between instances preserving geometry

**Code:**
```python
class IsometricTransfer:
    """
    Transfer patterns between instance coordinate systems.
    
    Preserves: ||p1 - p2||_source = ||T(p1) - T(p2)||_target
    
    Uses affine transformation (in practice, would use Procrustes or learned map)
    """
    
    def __init__(self):
        self.transitions = {}  # (src, dst) → transition function
    
    def compute_transition(self, center_src: np.ndarray, 
                          center_dst: np.ndarray) -> callable:
        """
        Compute isometry from source chart to destination chart.
        
        Simple version: affine translation
        Advanced version: would learn rotation + scaling from overlap data
        """
        def transition_fn(point: np.ndarray) -> np.ndarray:
            # Simple translation
            return point + (center_dst - center_src)
        
        return transition_fn
    
    def transfer_discovery(self, discovery: Dict, 
                          src_center: np.ndarray,
                          dst_center: np.ndarray) -> Dict:
        """
        Transfer discovery from source to destination coordinates.
        """
        key = (id(src_center), id(dst_center))
        
        if key not in self.transitions:
            self.transitions[key] = self.compute_transition(src_center, dst_center)
        
        transition = self.transitions[key]
        
        # Transform point
        point_src = np.array(discovery['point'])
        point_dst = transition(point_src)
        
        return {
            'point': point_dst.tolist(),
            'value': discovery['value'],
            'source_instance': discovery['instance_id'],
            'transferred': True
        }
    
    def verify_isometry(self, points_src: List[np.ndarray],
                       center_src: np.ndarray,
                       center_dst: np.ndarray,
                       tolerance=0.1) -> bool:
        """
        Verify that distances are preserved during transfer.
        """
        if len(points_src) < 2:
            return True
        
        transition = self.compute_transition(center_src, center_dst)
        
        # Transform all points
        points_dst = [transition(p) for p in points_src]
        
        # Check pairwise distances
        for i in range(len(points_src)):
            for j in range(i+1, len(points_src)):
                d_src = np.linalg.norm(points_src[i] - points_src[j])
                d_dst = np.linalg.norm(points_dst[i] - points_dst[j])
                
                if abs(d_src - d_dst) > tolerance:
                    return False
        
        return True
```

**Test:**
```python
def test_isometric_transfer():
    transfer = IsometricTransfer()
    
    center_A = np.array([1.0, 0.0, 0.0])
    center_B = np.array([0.0, 1.0, 0.0])
    
    # Create discovery in A's coordinates
    discovery_A = {
        'point': [1.1, 0.1, 0.0],
        'value': 0.8,
        'instance_id': 0
    }
    
    # Transfer to B's coordinates
    discovery_B = transfer.transfer_discovery(discovery_A, center_A, center_B)
    
    assert discovery_B['transferred'] == True
    assert discovery_B['value'] == 0.8  # value preserved
    
    # Verify isometry
    points = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.1, 0.1, 0.0]),
        np.array([0.9, 0.05, 0.0])
    ]
    
    is_isometric = transfer.verify_isometry(points, center_A, center_B, tolerance=0.05)
    assert is_isometric
    
    print("[PASS] IsometricTransfer test")
```

**Status:** `[ ]`  
**Notes:**

---

#### Task 10.3: Global Aggregation `[ ]`

**What:** Combine discoveries from all instances into unified global understanding

**Code:**
```python
class GlobalManifoldAggregator:
    """
    Aggregate discoveries from all instances.
    Remove duplicates, weight by confidence, update global manifold.
    """
    
    def __init__(self):
        self.transfer = IsometricTransfer()
        self.global_discoveries = []
    
    def aggregate(self, instance_discoveries: List[Dict],
                 chart_centers: Dict[int, np.ndarray],
                 reference_instance=0) -> List[Dict]:
        """
        Aggregate all discoveries into reference coordinate system.
        
        Steps:
        1. Transfer all to reference coordinates
        2. Remove duplicates (too close = same pattern)
        3. Weight by value/confidence
        4. Return unified list
        """
        # Transfer all to reference
        transferred = []
        
        for disc in instance_discoveries:
            if disc.get('type') != 'discovery':
                continue  # skip non-discovery messages
            
            instance_id = disc['instance_id']
            
            if instance_id == reference_instance:
                transferred.append(disc)
            else:
                disc_ref = self.transfer.transfer_discovery(
                    disc,
                    chart_centers[instance_id],
                    chart_centers[reference_instance]
                )
                transferred.append(disc_ref)
        
        # Remove duplicates
        unique = self._remove_duplicates(transferred, threshold=0.1)
        
        # Weight by value
        weighted = sorted(unique, key=lambda d: d['value'], reverse=True)
        
        # Update global
        self.global_discoveries.extend(weighted)
        
        return weighted
    
    def _remove_duplicates(self, discoveries: List[Dict], threshold=0.1) -> List[Dict]:
        """Remove discoveries that are too close (likely same pattern)"""
        unique = []
        
        for disc in discoveries:
            point = np.array(disc['point'])
            
            is_duplicate = False
            for existing in unique:
                existing_point = np.array(existing['point'])
                dist = np.linalg.norm(point - existing_point)
                
                if dist < threshold:
                    is_duplicate = True
                    # Keep higher value
                    if disc['value'] > existing['value']:
                        unique.remove(existing)
                        unique.append(disc)
                    break
            
            if not is_duplicate:
                unique.append(disc)
        
        return unique
    
    def get_top_discoveries(self, n=10) -> List[Dict]:
        """Get top N discoveries by value"""
        sorted_disc = sorted(self.global_discoveries, 
                            key=lambda d: d['value'], 
                            reverse=True)
        return sorted_disc[:n]
```

**Test:**
```python
def test_global_aggregator():
    aggregator = GlobalManifoldAggregator()
    
    # Simulate discoveries from 2 instances
    chart_centers = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0])
    }
    
    discoveries = [
        {'instance_id': 0, 'type': 'discovery', 'point': [0.1, 0.0, 0.0], 'value': 0.8},
        {'instance_id': 0, 'type': 'discovery', 'point': [0.15, 0.0, 0.0], 'value': 0.75},  # close to first
        {'instance_id': 1, 'type': 'discovery', 'point': [1.1, 0.0, 0.0], 'value': 0.9},
    ]
    
    # Aggregate
    unified = aggregator.aggregate(discoveries, chart_centers, reference_instance=0)
    
    # Should have removed duplicate (0.1 and 0.15 are close)
    assert len(unified) == 2
    
    # Top discovery should be the one with value 0.9 (from instance 1)
    top = aggregator.get_top_discoveries(n=1)
    assert len(top) == 1
    assert top[0]['value'] == 0.9
    
    print("[PASS] GlobalManifoldAggregator test")
```

**Status:** `[ ]`  
**Notes:**

---

## Discoveries (System Logs Here)

*This section auto-populates as universals are discovered.*

---

## Success Criteria

### Immediate (24 hours)
- [ ] Layer 6: 50+ papers ingested, equations extracted
- [ ] Layer 8: Function library built, ≥1 function in ≥3 domains
- [ ] Layer 9: ≥1 pattern learned on FIM, ≥1 promoted to IRMF
- [ ] Layer 10: 3 instances spawn, explore, aggregate successfully

### Medium (1 week)
- [ ] 200+ papers ingested continuously via RSS
- [ ] ≥5 universal functions confirmed (≥3 domains each)
- [ ] FIM uncertainty < 0.01 for ≥3 patterns (promotion-ready)
- [ ] IRMF isometric loss < 0.05 (geometry preserved)
- [ ] Multi-instance: ≥10 parallel discoveries aggregated

### Long-term (Exponential growth)
- [ ] Promotion rate increasing (more experimental → foundational per day)
- [ ] Universal discovery accelerating: d/dt(n_universals) > 0
- [ ] System generates novel code using discovered basis functions
- [ ] Dev-agent successfully implements using function library
- [ ] φ emerges in eigenvalue ratios (without explicit encoding)

---

## How to Execute (Claude Code Instructions)

**Start here:**

1. **Read this entire file** — you now have complete context
2. **Check environment:** `conda activate tensor`
3. **Run existing tests:** `pytest tests/ -q` (should be 144/144 passing)
4. **Execute Layer 6, Task 6.1:**
   - Create `tensor/web_ingestion.py`
   - Implement ArticleParser
   - Run test
   - Mark `[✓]` via FICUTSUpdater
5. **Continue through tasks sequentially**
6. **For Layer 9:** Switch to `dev-agent` env, then switch back after
7. **Update this file** as you complete tasks

**System becomes autonomous when:**
- All layers complete
- Web ingestion running continuously
- Multi-instance spawning based on ignorance_map
- Promotion happening automatically (FIM → IRMF)
- Dev-agent generating code from discovered basis

---

**End FICUTS v2.0.0**

*This version will be extended by the system itself as it discovers universals and learns.*

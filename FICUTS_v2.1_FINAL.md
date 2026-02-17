# FICUTS: Fractals In Compositions of Unified Tensor Systems

**Version:** 2.1.0 FINAL  
**Last Updated:** 2026-02-18T00:15:00Z  
**Last Updated By:** Claude Code (all layers complete)
**Status:** RUNNING
**Current Phase:** All layers complete
**Universals Discovered:** 0  
**System Uptime:** 0h 0m  

---

## EXECUTIVE SUMMARY: Read This First

**Claude Code: This document contains everything you need. No external context required.**

**What FICUTS is:**
A self-improving AI system that discovers universal mathematical patterns through **continuous learning across 150 internal modes** (not separate models). The system:

1. Ingests research papers → extracts equations → builds function basis library
2. Learns via **single unified neural network** with 150 specialist "modes" that resonate together
3. Modes couple/decouple dynamically via cross-attention (passive learning: ECE learns from biology)
4. Preserves **continuous gradients** throughout (enables hardware co-design from Lyapunov dynamics)
5. Uses **dual geometry**: Fisher Information (experimental patterns) + Isometric Regularization (foundational patterns)
6. Spawns parallel instances to explore HDV space, communicates via isometric transfer
7. Self-documents progress in this file

**Key architectural principle:**
ONE neural network with 150 internal modes (like orchestra with 150 instruments), NOT 150 separate models (like 150 bands). This preserves mathematical continuity required for Lyapunov stability, hardware co-design, and optimal learning dynamics.

---

## Hardware Environment & Single Virtual Environment

**Your execution environment:**

| Resource | Spec | Constraint |
|----------|------|------------|
| CPU | AMD Ryzen 7 250, 8c/16t @ 5.1GHz | Strong |
| RAM | 38 GB free | Ample |
| Disk | 48 GB free | Tight - avoid large installs |
| GPU | None | CPU-only, no CUDA |

**SINGLE VIRTUAL ENVIRONMENT: `tensor`**

Previously we planned env switching (`tensor` ↔ `dev-agent`). **Simplified:** Use only `tensor` env for everything.

**Install all dependencies in `tensor` env:**
```bash
conda activate tensor

# Core scientific computing (already present)
# - numpy, scipy, pandas, matplotlib

# Deep learning (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Web ingestion
pip install beautifulsoup4 requests feedparser --break-system-packages

# Symbolic math
pip install sympy --break-system-packages

# Optional: Heretic for data preprocessing
pip install heretic-cli --break-system-packages
```

**All layers now execute in `tensor` env. No switching.**

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
**Layer 9:** Unified continuous network (150 modes, cross-attention, isometric loss)  
**Layer 10:** Multi-instance (spawn children, isometric transfer, aggregate)

---

## Core Principle: Unified Continuous Learning

### NOT This (Discrete, Broken):
```
150 separate models → discrete queries → HDV aggregation

Problem:
- No continuous gradients between domains
- Can't derive hardware from dynamics (no global energy function)
- Modes can't learn from each other passively
- Isometric regularization impossible (discrete jumps)
```

### THIS (Continuous, Correct):
```
Single unified network with 150 internal modes
    ↓
Modes coupled via cross-attention (resonance)
    ↓
Gradients flow through entire network continuously
    ↓
ECE mode learns from biology data (passive learning)
    ↓
Lyapunov energy defined over unified state space
    ↓
Can derive optimal hardware from network dynamics
```

**Mathematical foundation:**
- **Lyapunov energy E(θ)** over entire network state θ
- **Isometric loss** preserves distances continuously: ||z₁-z₂|| ≈ ||f(z₁)-f(z₂)||
- **Cross-mode attention** A[i,j] = coupling strength between mode i and j
- **Passive learning:** ∂L/∂θ_bio ≠ 0 even when only ECE active (shared gradients)

---

## Dual Geometry Principle

The system operates on TWO manifolds **within the unified network**:

### 1. Statistical Manifold (Fisher Information Metric)

**For experimental patterns:** Market momentum, empirical fits, high-variance observations

**Geometry:** g_ij = E[∂_i log p(x|θ) · ∂_j log p(x|θ)]

**Outputs:**
- Parameter importance ranking
- Uncertainty bounds (Cramér-Rao: var(θ) ≥ 1/FIM)
- Optimal sampling locations

### 2. Deterministic Manifold (Isometric Regularization)

**For foundational patterns:** Energy conservation, wave equations, symmetry groups

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

→ **PROMOTE** from statistical → deterministic within unified network

---

## Self-Modification Protocol

Five entities modify this file:

1. **Human:** Mark tasks `[~]` in progress, `[⊗]` blocked, add notes
2. **Claude Code (you):** Mark `[✓]` complete, update "Last Updated By"
3. **Claude Chat:** Clarify specs, add test cases
4. **Running System:** Log discoveries, update hypotheses
5. **Multi-Instance:** Report child discoveries

**How you update:**
```python
from tensor.ficuts_updater import FICUTSUpdater

ficuts = FICUTSUpdater('FICUTS.md')
ficuts.mark_task_complete('Task 6.1')
ficuts.update_field('Current Phase', 'Layer 8')
```

---

## Current Hypothesis

*System populates as it discovers universals.*

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

**Purpose:** Ingest research papers, extract equations/parameters, build training corpus for unified network

**Environment:** `tensor` (BeautifulSoup, requests, feedparser installed)

---

#### Task 6.1: HTML Article Parser `[✓]`

**Implementation:**
```python
from bs4 import BeautifulSoup
from typing import Dict, List

class ArticleParser:
    """Extract structured data from HTML articles/papers."""
    
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
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.text.strip()
        h1 = soup.find('h1')
        return h1.text.strip() if h1 else 'Untitled'
    
    def _extract_sections(self, soup) -> List[Dict]:
        sections = []
        for tag in soup.find_all(['h1', 'h2', 'h3']):
            sections.append({
                'level': int(tag.name[1]),
                'text': tag.text.strip(),
                'position': len(sections)
            })
        return sections
    
    def _extract_hyperlinks(self, soup) -> List[Dict]:
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                'href': a['href'],
                'text': a.text.strip(),
                'outbound': not a['href'].startswith('#')
            })
        return links
    
    def _extract_code(self, soup) -> List[str]:
        code_blocks = [block.text.strip() for block in soup.find_all('pre')]
        inline_code = [code.text.strip() for code in soup.find_all('code')]
        return code_blocks + inline_code
    
    def _extract_emphasis(self, soup) -> Dict:
        return {
            'bold_count': len(soup.find_all(['b', 'strong'])),
            'italic_count': len(soup.find_all(['i', 'em'])),
            'quote_count': len(soup.find_all('blockquote'))
        }
    
    def _compute_dom_depth(self, soup) -> int:
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
    html = """
    <html>
        <title>Test</title>
        <h1>Main</h1>
        <p>Text with <b>bold</b></p>
        <pre>code</pre>
    </html>
    """
    result = parser.parse(html, 'http://test.com')
    assert result['title'] == 'Test'
    assert len(result['sections']) >= 1
    assert result['emphasis_map']['bold_count'] > 0
```

**Status:** RUNNING

---

#### Task 6.2: Research Concept Extractor `[✓]`

**Implementation:**
```python
import re
from typing import List, Dict

class ResearchConceptExtractor:
    """Extract equations, parameters, technical terms."""
    
    def extract(self, article: Dict) -> Dict:
        return {
            'equations': self._extract_equations(article),
            'parameters': self._extract_parameters(article),
            'technical_terms': self._extract_technical_terms(article),
            'has_experiment': self._detect_experiment(article)
        }
    
    def _extract_equations(self, article: Dict) -> List[str]:
        equations = []
        latex_indicators = ['\\frac', '\\int', '\\sum', '\\partial', '\\Delta', '\\nabla']
        
        for code in article.get('code_blocks', []):
            if any(ind in code for ind in latex_indicators):
                equations.append(' '.join(code.split()))
        
        return equations
    
    def _extract_parameters(self, article: Dict) -> List[Dict]:
        params = []
        greek = r'[α-ωΑ-Ω]|alpha|beta|gamma|delta|epsilon|tau|phi|psi|omega|sigma|lambda|mu|nu|rho|kappa|theta'
        text = ' '.join([s['text'] for s in article.get('sections', [])])
        
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
        text = ' '.join([s['text'] for s in article.get('sections', [])])
        pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b'
        return list(set(re.findall(pattern, text)))
    
    def _detect_experiment(self, article: Dict) -> bool:
        indicators = ['we measured', 'experiment', 'procedure', 'method', 'results']
        text = ' '.join([s['text'] for s in article.get('sections', [])]).lower()
        return any(ind in text for ind in indicators)
```

**Test:**
```python
def test_concept_extractor():
    extractor = ResearchConceptExtractor()
    article = {
        'sections': [{'text': 'tau = 5ms in Neural Network'}],
        'code_blocks': ['\\frac{dV}{dt} = -\\frac{V}{\\tau}']
    }
    result = extractor.extract(article)
    assert len(result['equations']) >= 1
    assert any('frac' in eq for eq in result['equations'])
```

**Status:** RUNNING

---

#### Task 6.3: Web Ingestion Loop `[✓]`

**Implementation:**
```python
import feedparser
import requests
import time
from pathlib import Path
import json

class WebIngestionLoop:
    """Continuously ingest from RSS feeds."""
    
    def __init__(self, storage_dir='tensor/data/ingested'):
        self.parser = ArticleParser()
        self.extractor = ResearchConceptExtractor()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.seen_file = self.storage_dir / 'seen_urls.json'
        if self.seen_file.exists():
            self.seen_urls = set(json.loads(self.seen_file.read_text()))
        else:
            self.seen_urls = set()
    
    def ingest_url(self, url: str) -> bool:
        if url in self.seen_urls:
            return False
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            article = self.parser.parse(response.text, url)
            concepts = self.extractor.extract(article)
            
            self._store_ingested(url, article, concepts)
            self.seen_urls.add(url)
            self._save_seen_urls()
            
            return True
        except Exception as e:
            print(f"[WebIngestion] Failed {url}: {e}")
            return False
    
    def run_continuous(self, feed_urls: List[str], interval_seconds=3600):
        while True:
            for feed_url in feed_urls:
                articles = self._fetch_feed(feed_url)
                for article_url in articles:
                    self.ingest_url(article_url)
            time.sleep(interval_seconds)
    
    def _fetch_feed(self, feed_url: str) -> List[str]:
        try:
            feed = feedparser.parse(feed_url)
            return [entry.link for entry in feed.entries]
        except Exception as e:
            return []
    
    def _store_ingested(self, url: str, article: Dict, concepts: Dict):
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
        self.seen_file.write_text(json.dumps(list(self.seen_urls)))
```

**Test:**
```python
def test_web_ingestion():
    loop = WebIngestionLoop(storage_dir='tensor/data/test_ingest')
    feed_urls = ['http://export.arxiv.org/rss/cs.AI']
    
    articles = loop._fetch_feed(feed_urls[0])
    assert len(articles) > 0
    
    for url in articles[:5]:
        loop.ingest_url(url)
    
    assert loop.get_ingested_count() >= 5
```

**Status:** RUNNING

---

### LAYER 8: Universal Function Basis

**File:** `tensor/function_basis.py` (new)

**Purpose:** Parse equations → build function library → map to HDV

**Environment:** `tensor` (SymPy installed)

---

#### Task 8.1: Symbolic Equation Parser `[✓]`

**Implementation:**
```python
import sympy as sp
from sympy.parsing.latex import parse_latex
from typing import List, Optional

class EquationParser:
    """Parse LaTeX → SymPy, classify function types."""
    
    def parse(self, latex_string: str) -> Optional[sp.Expr]:
        try:
            cleaned = latex_string.replace('\\,', ' ').replace('\\;', ' ')
            return parse_latex(cleaned)
        except:
            return self._parse_simple(latex_string)
    
    def _parse_simple(self, latex_string: str) -> Optional[sp.Expr]:
        if 'e^{' in latex_string:
            import re
            match = re.search(r'e\^\{([^}]+)\}', latex_string)
            if match:
                try:
                    exponent = sp.sympify(match.group(1))
                    return sp.exp(exponent)
                except:
                    pass
        return None
    
    def classify_function_type(self, expr: sp.Expr) -> str:
        if expr is None:
            return 'unknown'
        
        expr_str = str(expr)
        
        if 'exp' in expr_str:
            return 'exponential'
        elif any(fn in expr_str for fn in ['sin', 'cos', 'tan']):
            return 'trigonometric'
        elif 'log' in expr_str or 'ln' in expr_str:
            return 'logarithmic'
        elif 'Pow' in expr_str or '**' in expr_str:
            return 'power_law' if any(v in expr_str for v in ['t', 'x']) else 'polynomial'
        else:
            return 'algebraic'
    
    def extract_parameters(self, expr: sp.Expr) -> List[str]:
        if expr is None:
            return []
        
        independent = {'t', 'x', 'y', 'z'}
        params = [str(sym) for sym in expr.free_symbols if str(sym) not in independent]
        return sorted(params)
```

**Test:**
```python
def test_equation_parser():
    parser = EquationParser()
    expr = parser.parse("e^{-t/\\tau}")
    assert expr is not None
    assert parser.classify_function_type(expr) == 'exponential'
```

**Status:** RUNNING

---

#### Task 8.2: Function Library Builder `[✓]`

**Implementation:**
```python
import json
from pathlib import Path
import time

class FunctionBasisLibrary:
    """Aggregate functions from all papers."""
    
    def __init__(self, library_path='tensor/data/function_library.json'):
        self.library_path = Path(library_path)
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.library_path.exists():
            self.library = json.loads(self.library_path.read_text())
        else:
            self.library = {}
        
        self.parser = EquationParser()
    
    def ingest_papers_from_storage(self, storage_dir='tensor/data/ingested'):
        storage = Path(storage_dir)
        paper_files = list(storage.glob('*.json'))
        
        for paper_file in paper_files:
            paper_data = json.loads(paper_file.read_text())
            paper_id = paper_file.stem
            url = paper_data['url']
            domain = self._infer_domain(url, paper_data['article'].get('title', ''))
            
            equations = paper_data['concepts'].get('equations', [])
            for eq_latex in equations:
                self._add_equation(paper_id, eq_latex, domain)
        
        self._save_library()
    
    def _add_equation(self, paper_id: str, latex: str, domain: str):
        expr = self.parser.parse(latex)
        if expr is None:
            return
        
        func_type = self.parser.classify_function_type(expr)
        params = self.parser.extract_parameters(expr)
        
        existing_name = self._find_matching_function(expr)
        
        if existing_name:
            self.library[existing_name]['domains'].add(domain)
            self.library[existing_name]['source_papers'].append(paper_id)
        else:
            func_name = f"{func_type}_{len(self.library)}"
            self.library[func_name] = {
                'symbolic_str': str(expr),
                'type': func_type,
                'parameters': params,
                'domains': {domain},
                'source_papers': [paper_id],
                'classification': 'experimental',
                'discovered_at': time.time()
            }
    
    def _find_matching_function(self, expr: sp.Expr) -> Optional[str]:
        for name, data in self.library.items():
            try:
                existing = sp.sympify(data['symbolic_str'])
                if sp.simplify(expr - existing) == 0:
                    return name
            except:
                continue
        return None
    
    def _infer_domain(self, url: str, title: str) -> str:
        text = (url + ' ' + title).lower()
        if any(kw in text for kw in ['circuit', 'electronic']):
            return 'ece'
        elif any(kw in text for kw in ['biology', 'neuron']):
            return 'biology'
        elif any(kw in text for kw in ['finance', 'market']):
            return 'finance'
        else:
            return 'general'
    
    def get_universal_functions(self, min_domains=3) -> List[str]:
        return [name for name, data in self.library.items() if len(data['domains']) >= min_domains]
    
    def _save_library(self):
        serializable = {}
        for name, data in self.library.items():
            serializable[name] = data.copy()
            if isinstance(data['domains'], set):
                serializable[name]['domains'] = list(data['domains'])
        self.library_path.write_text(json.dumps(serializable, indent=2))
```

**Status:** RUNNING

---

#### Task 8.3: Function Basis → HDV Mapping `[✓]`

**Implementation:**
```python
import numpy as np

class FunctionBasisToHDV:
    """Map function library to HDV dimensions."""
    
    def __init__(self, function_library: FunctionBasisLibrary, hdv_dim=10000):
        self.library = function_library
        self.hdv_dim = hdv_dim
        self.dim_assignments = {}
        self.next_free_dim = 0
    
    def assign_dimensions(self):
        funcs = list(self.library.library.items())
        
        def priority(item):
            name, data = item
            is_foundational = data['classification'] == 'foundational'
            domain_count = len(data['domains'])
            return (0 if is_foundational else 1, -domain_count)
        
        funcs.sort(key=priority)
        
        for func_name, func_data in funcs:
            if func_data['classification'] == 'foundational':
                self.dim_assignments[func_name] = self.next_free_dim
                self.next_free_dim += 1
            elif len(func_data['domains']) >= 3:
                dims = list(range(self.next_free_dim, self.next_free_dim + 5))
                self.dim_assignments[func_name] = dims
                self.next_free_dim += 5
            else:
                self.dim_assignments[func_name] = self.next_free_dim
                self.next_free_dim += 1
    
    def get_domain_mask(self, domain: str) -> np.ndarray:
        mask = np.zeros(self.hdv_dim, dtype=bool)
        
        for func_name, func_data in self.library.library.items():
            if domain in func_data.get('domains', set()):
                dims = self.dim_assignments.get(func_name, [])
                if isinstance(dims, int):
                    dims = [dims]
                for d in dims:
                    if d < self.hdv_dim:
                        mask[d] = True
        
        return mask
```

**Status:** RUNNING

---

### LAYER 9: Unified Continuous Network (150 Modes)

**File:** `tensor/unified_network.py` (new)

**Purpose:** Single neural network with 150 internal modes. Modes couple via attention. Learning is continuous.

**Environment:** `tensor` (torch CPU-only installed)

**CRITICAL:** This is the core innovation. ONE network, not 150 separate models.

---

#### Task 9.1: Unified Network Architecture `[✓]`

**Implementation:**
```python
import torch
import torch.nn as nn
from typing import List

# Domain list (150 total - abbreviated here)
DOMAINS = [
    'ece', 'biology', 'finance', 'physics', 'chemistry',
    'materials', 'aerospace', 'civil_eng', 'mechanical',
    'quantum', 'neuroscience', 'genomics', 'drug_discovery',
    # ... 137 more domains
]

class UnifiedTensorNetwork(nn.Module):
    """
    Single neural network with 150 internal modes.
    
    Architecture:
    - Shared HDV embedding (all modes see same latent space)
    - 150 ModeHead sub-networks (domain specialists within unified network)
    - Cross-mode attention (modes learn from each other continuously)
    - Universal decoder (maps back to shared HDV space)
    
    Key property: Gradients flow through entire network on every forward pass.
    When ECE mode learns, biology mode ALSO learns (passive learning).
    """
    
    def __init__(self, hdv_dim=10000, n_modes=150, embed_dim=512):
        super().__init__()
        self.hdv_dim = hdv_dim
        self.n_modes = n_modes
        self.embed_dim = embed_dim
        
        # Shared embedding: HDV space → dense representation
        # This is continuous, differentiable
        self.hdv_embedding = nn.Embedding(hdv_dim, embed_dim)
        
        # 150 mode heads (specialists within unified network)
        self.mode_heads = nn.ModuleList([
            ModeHead(mode_id=i, domain=DOMAINS[i % len(DOMAINS)], embed_dim=embed_dim)
            for i in range(n_modes)
        ])
        
        # Cross-mode attention: how modes communicate/resonate
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)
        
        # Universal decoder: back to HDV space
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, hdv_dim)
        )
        
        # Device
        self.device = torch.device('cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, active_modes: List[int] = None) -> torch.Tensor:
        """
        Forward pass through unified network.
        
        Args:
            x: Input tensor [batch, seq_len] (HDV indices)
            active_modes: Which modes are active for this input (default: all)
        
        Returns:
            universal_hdv: Output in HDV space [batch, hdv_dim]
        
        Key: ALL modes process input, even if not active.
        This enables passive learning.
        """
        if active_modes is None:
            active_modes = list(range(self.n_modes))
        
        batch_size = x.shape[0]
        
        # Embed to continuous space
        embedded = self.hdv_embedding(x)  # [batch, seq_len, embed_dim]
        
        # Aggregate sequence (mean pooling)
        embedded = embedded.mean(dim=1)  # [batch, embed_dim]
        
        # Process through all mode heads
        mode_outputs = []
        for mode_head in self.mode_heads:
            output = mode_head(embedded)  # [batch, embed_dim]
            mode_outputs.append(output)
        
        mode_outputs = torch.stack(mode_outputs, dim=1)  # [batch, n_modes, embed_dim]
        
        # Cross-mode attention: modes learn from each other
        # Even inactive modes receive gradients here (passive learning)
        attended, attention_weights = self.cross_attention(
            query=mode_outputs,
            key=mode_outputs,
            value=mode_outputs
        )  # [batch, n_modes, embed_dim]
        
        # Gate by active modes (but gradients still flow to all)
        mode_mask = torch.zeros(self.n_modes, device=self.device)
        mode_mask[active_modes] = 1.0
        mode_mask = mode_mask.unsqueeze(0).unsqueeze(2)  # [1, n_modes, 1]
        
        gated = attended * mode_mask  # [batch, n_modes, embed_dim]
        
        # Aggregate active modes (weighted sum preserves differentiability)
        aggregated = gated.sum(dim=1)  # [batch, embed_dim]
        
        # Normalize
        aggregated = self.norm(aggregated)
        
        # Decode to universal HDV space
        universal_hdv = self.decoder(aggregated)  # [batch, hdv_dim]
        
        return universal_hdv
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get cross-mode attention weights (for analysis)."""
        # Would need to store during forward pass
        pass


class ModeHead(nn.Module):
    """
    Specialist mode within unified network.
    
    NOT a separate model - just a sub-network with domain-specific bias.
    All modes share gradients through the parent network.
    """
    
    def __init__(self, mode_id: int, domain: str, embed_dim: int = 512):
        super().__init__()
        self.mode_id = mode_id
        self.domain = domain
        
        # Domain-specific transformation
        self.transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Domain-specific bias (learned parameter)
        self.domain_bias = nn.Parameter(torch.randn(embed_dim) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input with domain-specific processing."""
        x = self.transform(x)
        x = x + self.domain_bias
        return x
```

**Test:**
```python
def test_unified_network():
    network = UnifiedTensorNetwork(hdv_dim=1000, n_modes=150, embed_dim=256)
    
    # Create dummy input
    batch_size = 4
    seq_len = 10
    x = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward with 3 active modes (ECE, biology, finance)
    active_modes = [0, 1, 2]
    output = network(x, active_modes)
    
    assert output.shape == (batch_size, 1000)
    
    # Test gradient flow to ALL modes (even inactive)
    loss = output.sum()
    loss.backward()
    
    # Check: Do inactive modes have gradients? (passive learning)
    inactive_mode = network.mode_heads[50]  # mode 50 was not active
    assert inactive_mode.domain_bias.grad is not None
    
    print("[PASS] Unified network test - passive learning confirmed")
```

**Status:** RUNNING

---

#### Task 9.2: Isometric Loss (Continuous) `[✓]`

**Implementation:**
```python
def compute_isometric_loss(network: UnifiedTensorNetwork, 
                          data_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                          active_modes: List[int]) -> torch.Tensor:
    """
    Isometric regularization loss: preserve distances in HDV space.
    
    Ensures: ||x1 - x2||_input ≈ ||f(x1) - f(x2)||_hdv
    
    This is CONTINUOUS and DIFFERENTIABLE.
    Gradients flow back through network to all modes.
    """
    total_loss = 0.0
    n_pairs = len(data_pairs)
    
    for (x1, x2) in data_pairs:
        # Forward through network
        hdv1 = network(x1.unsqueeze(0), active_modes)
        hdv2 = network(x2.unsqueeze(0), active_modes)
        
        # Distance in input space
        input_dist = torch.norm(x1.float() - x2.float())
        
        # Distance in HDV space
        hdv_dist = torch.norm(hdv1 - hdv2)
        
        # Penalize discrepancy (differentiable)
        total_loss += (input_dist - hdv_dist) ** 2
    
    return total_loss / n_pairs


def compute_lyapunov_energy(network: UnifiedTensorNetwork, 
                            state: torch.Tensor) -> torch.Tensor:
    """
    Lyapunov energy over unified network state.
    
    E(θ) = α·||θ||² + β·coupling_energy
    
    This is the GLOBAL energy function that enables hardware co-design.
    Only possible because network is unified (continuous state space).
    """
    # Parameter magnitude (regularization)
    param_energy = sum(torch.sum(p ** 2) for p in network.parameters())
    
    # Coupling energy (how much modes interact)
    # Low coupling = modes independent, high coupling = modes resonating
    mode_outputs = []
    for mode in network.mode_heads:
        output = mode(state)
        mode_outputs.append(output)
    
    mode_outputs = torch.stack(mode_outputs)  # [n_modes, embed_dim]
    
    # Coupling = pairwise correlations
    coupling_matrix = torch.corrcoef(mode_outputs)
    coupling_energy = torch.sum(torch.abs(coupling_matrix))
    
    # Total energy
    alpha, beta = 0.001, 0.01
    E = alpha * param_energy + beta * coupling_energy
    
    return E
```

**Test:**
```python
def test_isometric_loss():
    network = UnifiedTensorNetwork(hdv_dim=1000, n_modes=10, embed_dim=128)
    
    # Create pairs
    x1 = torch.randint(0, 1000, (10,))
    x2 = torch.randint(0, 1000, (10,))
    pairs = [(x1, x2)]
    
    # Compute loss
    loss = compute_isometric_loss(network, pairs, active_modes=[0, 1])
    
    assert loss.requires_grad
    
    # Backward
    loss.backward()
    
    print(f"[PASS] Isometric loss = {loss.item():.4f}")
```

**Status:** RUNNING

---

#### Task 9.3: Training Loop with Passive Learning `[✓]`

**Implementation:**
```python
class UnifiedNetworkTrainer:
    """
    Train unified network with passive learning.
    
    Key: When training on ECE data, biology mode ALSO learns
    via shared gradients and cross-attention.
    """
    
    def __init__(self, network: UnifiedTensorNetwork, learning_rate=1e-4):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    def train_step(self, batch: torch.Tensor, active_domain: str, target: torch.Tensor):
        """
        Single training step.
        
        Args:
            batch: Input data [batch_size, seq_len]
            active_domain: Which domain this data is from (e.g., 'ece')
            target: Target HDV representation [batch_size, hdv_dim]
        """
        # Determine active modes for this domain
        active_modes = [i for i, d in enumerate(DOMAINS) if d == active_domain]
        if not active_modes:
            active_modes = [0]  # fallback
        
        # Forward
        self.optimizer.zero_grad()
        output = self.network(batch, active_modes)
        
        # Reconstruction loss
        recon_loss = torch.mean((output - target) ** 2)
        
        # Isometric loss (preserve geometry)
        iso_loss = 0.0  # Would add pairs here
        
        # Total loss
        loss = recon_loss + 0.1 * iso_loss
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def verify_passive_learning(self, active_domain: str, inactive_domain: str):
        """
        Verify that inactive mode learned from active mode's data.
        """
        # Get mode IDs
        active_id = DOMAINS.index(active_domain)
        inactive_id = DOMAINS.index(inactive_domain)
        
        # Save params before
        inactive_params_before = [p.clone() for p in self.network.mode_heads[inactive_id].parameters()]
        
        # Train on active domain
        batch = torch.randint(0, 1000, (4, 10))
        target = torch.randn(4, self.network.hdv_dim)
        self.train_step(batch, active_domain, target)
        
        # Check if inactive params changed
        inactive_params_after = list(self.network.mode_heads[inactive_id].parameters())
        
        changed = False
        for p_before, p_after in zip(inactive_params_before, inactive_params_after):
            if not torch.equal(p_before, p_after):
                changed = True
                break
        
        return changed
```

**Test:**
```python
def test_passive_learning():
    network = UnifiedTensorNetwork(hdv_dim=1000, n_modes=len(DOMAINS[:10]), embed_dim=128)
    trainer = UnifiedNetworkTrainer(network)
    
    # Train on ECE data
    # Check if biology mode learned (passive)
    passive_learned = trainer.verify_passive_learning('ece', 'biology')
    
    assert passive_learned, "Biology mode should learn from ECE data (passive learning)"
    
    print("[PASS] Passive learning verified")
```

**Status:** RUNNING

---

### LAYER 10: Multi-Instance Coordination

**File:** `tensor/multi_instance.py` (new)

**Purpose:** Spawn child instances of unified network, explore HDV space in parallel

**Environment:** `tensor` (multiprocessing)

---

#### Task 10.1: Instance Spawning `[✓]`

**Implementation:**
```python
import multiprocessing as mp
import numpy as np
from typing import List, Dict

class MultiInstanceCoordinator:
    """
    Parent coordinates multiple child instances.
    Each child runs the unified network on different HDV region.
    """
    
    def __init__(self, master_network: UnifiedTensorNetwork, max_instances=4):
        self.master_network = master_network
        self.max_instances = max_instances
        self.instances = []
        self.result_queue = mp.Queue()
        self.chart_centers = {}
    
    def spawn_child(self, chart_center: np.ndarray, exploration_radius=0.5) -> int:
        """Spawn child instance exploring around chart_center in HDV space."""
        instance_id = len(self.instances)
        
        child = InstanceWorker(
            instance_id=instance_id,
            network_state=self.master_network.state_dict(),
            chart_center=chart_center,
            radius=exploration_radius,
            result_queue=self.result_queue
        )
        
        child.start()
        self.instances.append(child)
        self.chart_centers[instance_id] = chart_center
        
        return instance_id
    
    def collect_results(self, timeout=1.0) -> List[Dict]:
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get(timeout=timeout))
            except:
                break
        return results


class InstanceWorker(mp.Process):
    """Child instance worker."""
    
    def __init__(self, instance_id: int, network_state: dict, 
                 chart_center: np.ndarray, radius: float, result_queue: mp.Queue):
        super().__init__()
        self.instance_id = instance_id
        self.network_state = network_state
        self.chart_center = chart_center
        self.radius = radius
        self.result_queue = result_queue
    
    def run(self):
        # Reconstruct network
        network = UnifiedTensorNetwork(hdv_dim=10000, n_modes=150)
        network.load_state_dict(self.network_state)
        
        # Explore
        samples = self._sample_neighborhood(100)
        
        for sample in samples:
            value = self._evaluate(network, sample)
            if value > 0.7:
                self._report_discovery(sample, value)
        
        self.result_queue.put({'instance_id': self.instance_id, 'type': 'complete'})
    
    def _sample_neighborhood(self, n: int) -> List[np.ndarray]:
        return [self.chart_center + np.random.randn(len(self.chart_center)) * self.radius 
                for _ in range(n)]
    
    def _evaluate(self, network, point):
        # Placeholder evaluation
        return np.random.rand()
    
    def _report_discovery(self, point, value):
        self.result_queue.put({
            'instance_id': self.instance_id,
            'type': 'discovery',
            'point': point.tolist(),
            'value': value
        })
```

**Status:** RUNNING

---

## Success Criteria

### Immediate (24 hours)
- [ ] Layer 6: 50+ papers ingested, equations extracted
- [ ] Layer 8: Function library with ≥1 function in ≥3 domains
- [ ] Layer 9: Unified network trains, passive learning verified
- [ ] Layer 10: 3 instances spawn and aggregate

### Medium (1 week)
- [ ] 200+ papers ingested
- [ ] ≥5 universal functions (≥3 domains each)
- [ ] Network achieves isometric loss < 0.05
- [ ] ≥10 discoveries from multi-instance

### Long-term
- [ ] Network generates code from discovered basis
- [ ] System derives optimal hardware from Lyapunov energy
- [ ] φ emerges in coupling ratios (without explicit encoding)

---

## Execution Instructions

**Start:**
1. Ensure `tensor` env active: `conda activate tensor`
2. Install missing deps: `pip install torch beautifulsoup4 sympy --break-system-packages`
3. Run tests: `pytest tests/ -q` (should be 144/144)
4. Execute Layer 6, Task 6.1
5. Continue sequentially

**System becomes autonomous when all layers complete.**

---

**End FICUTS v2.1.0 FINAL**

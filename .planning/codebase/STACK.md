# Technology Stack

**Analysis Date:** 2026-02-22

## Languages

**Primary:**
- Python 3.11+ - Core platform, scientific computing, tensor analysis, code generation
- Rust - Performance-critical templates, compiled code generation, physics kernels
- JavaScript/TypeScript - Frontend UI (React)

**Secondary:**
- LaTeX - Equation extraction from academic papers
- YAML - Configuration in ecemath module

## Runtime

**Environment:**
- Python 3.11 (minimal requirement in `pyproject.toml`)
- Node.js with npm (Frontend build and dev server)

**Package Manager:**
- pip (Python dependencies)
- npm (JavaScript dependencies)
- conda (Development environment, specified in `platform/backend/start.sh`)

**Lockfiles:**
- `pyproject.toml` with setuptools build backend
- `package.json` for Node.js

## Frameworks

**Core Backend:**
- FastAPI 0.100+ - REST API framework for all 6 routers (`platform/backend/routers/`)
- Uvicorn 0.23+ - ASGI server for FastAPI deployment
- Pydantic 2.0+ - Request/response validation (BaseModel in all routers)

**Frontend:**
- React 18.2.0 - Component-based UI
- Vite 5.0.0 - Build tool and dev server (configured in `platform/frontend/vite.config.js`)
- Tailwind CSS 3.3.5 - Utility-first styling
- Lucide React 0.294.0 - Icon library

**Scientific Computing:**
- NumPy 1.24+ - Numerical arrays and linear algebra
- SciPy 1.11+ - Optimization, signal processing, eigenvalue problems
- scikit-learn 1.3+ - Machine learning utilities

**Optional Full Stack:**
- pandas 2.0+ - Data manipulation and DataFrames
- sympy 1.12+ - Symbolic mathematics
- torch (optional) - Used in `tensor/growing_network.py`, `tensor/integrated_hdv.py`, `tensor/unified_network.py` for neural networks

**Visualization (ecemath only):**
- matplotlib 3.4+ - Scientific plots
- plotly 5.0+ - Interactive plots
- networkx 2.6+ - Graph analysis

## Key Dependencies

**Critical:**
- requests - HTTP fetching for web ingestion (`tensor/deepwiki_integration.py`, `tensor/arxiv_pdf_parser.py`, `tensor/bootstrap_manager.py`)
- beautifulsoup4 - HTML parsing (guarded import, used in web ingestion and financial parsers)
- Scrapling (optional, preferred) - Advanced web scraping with Fetcher, StealthyFetcher, DynamicFetcher classes (`tensor/scrapling_ingestion.py`, `tensor/financial_ingestion.py`)

**Optimization & Hyperparameter Tuning:**
- optuna - Hyperparameter optimization with SQLite storage at `tensor/data/optuna_*.db` (used in `tensor/meta_optimizer.py`)

**Testing:**
- pytest 7.4+ - Test runner
- pytest-timeout 2.1+ - Timeout management for long tests
- pytest.ini in `pyproject.toml` with testpaths: `["tests", "platform/backend/tests"]`

**Code Generation & Compilation:**
- pyo3 - Python-Rust bindings for compiled templates (in `codegen/templates/numeric_kernel.py`)
- maturin (inferred) - Rust-Python build integration for code generation

## Configuration

**Environment:**
- PYTHONPATH must include project root and ecemath (`platform/backend/start.sh` pre-sets this)
- PYTHONUNBUFFERED=1 in Docker for live logging
- GITHUB_TOKEN (optional) for GitHub API access in `tensor/bootstrap_manager.py` and `tensor/deepwiki_navigator.py`

**Build:**
- `pyproject.toml` - setuptools configuration, package discovery (includes `tensor*`, `optimization*`, `codegen*`, `stability_engine*`)
- `platform/frontend/vite.config.js` - Vite proxy to `/api` → `http://localhost:8000`
- `platform/frontend/tailwind.config.js` - Custom colors (slate-850, slate-950) and JetBrains Mono font

**Data Storage:**
- SQLite (via Optuna) - `tensor/data/optuna_*.db` for study persistence
- JSON files - Configuration and state in `tensor/data/` (fed_dates.json, hdv_state.json, universals.json, function_library.json, ingestion_journal.json)
- Local filesystem - Ingested papers in `tensor/data/ingested/`, DeepWiki cache in `tensor/data/deepwiki_cache/`

## Platform Requirements

**Development:**
- conda environment named "tensor" with dependencies from `pyproject.toml` optional `full` extras
- Node.js 16+ for frontend development
- Python 3.11+ with pip
- Unix shell (bash) for startup scripts (`platform/backend/start.sh`)

**Production:**
- Docker container based on `python:3.11-slim` (see `stability_engine/Dockerfile`)
- Non-root user execution (appuser)
- Port 8000 exposed for FastAPI backend
- Health check endpoint at `/health` (HTTP GET)

## CI/CD

**Docker:**
- `docker-compose.yml` orchestrates `stability-engine` service on port 8000
- `.dockerignore` excludes monorepo modules, caches, and test artifacts
- `requirements-engine.txt` specifies production dependencies only (numpy, scipy, scikit-learn, fastapi, uvicorn, pydantic)

## Runtime Behavior

**Backend:**
- Starts with `uvicorn main:app --reload --port 8000` (development)
- Loads 6 FastAPI routers: regime, calendar, codegen, hdv, physics, circuit
- CORS middleware allows localhost:5173 (Vite dev) and localhost:3000 (production)
- All imports guarded with try/except for defensive operation (mock fallbacks available)

**Frontend:**
- Vite dev server on port 5173 (configurable via `npm run dev`)
- Proxies `/api` requests to backend at `http://localhost:8000`
- Build output: static files optimized with Vite

---

*Stack analysis: 2026-02-22*

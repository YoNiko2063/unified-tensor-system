# External Integrations

**Analysis Date:** 2026-02-22

## APIs & External Services

**Web Scraping & Information Retrieval:**
- Scrapling (optional) - Advanced web fetcher for site-specific HTML/JS scraping
  - SDK/Client: `scrapling.fetchers` (Fetcher, StealthyFetcher, DynamicFetcher, AsyncFetcher)
  - Usage: `tensor/scrapling_ingestion.py`, `tensor/financial_ingestion.py`
  - Fallback: requests.Session if unavailable
  - Auth: None (public APIs)

- arXiv - LaTeX source and paper metadata extraction
  - Endpoint: `https://arxiv.org/e-print/{paper_id}` (source archives)
  - Usage: `tensor/arxiv_pdf_parser.py`
  - Client: requests library
  - Auth: None (public API)
  - Rate limited: 3s delay between requests

- DeepWiki - AI-generated analysis of GitHub repositories
  - Endpoint: `https://deepwiki.com/{owner}/{repo}`
  - Usage: `tensor/deepwiki_integration.py`, `tensor/deepwiki_navigator.py`
  - Client: requests + BeautifulSoup4
  - Auth: Optional (no token required for basic access)
  - Cache: `tensor/data/deepwiki_cache/` (persists parsed workflows)

**GitHub Integration:**
- GitHub API v3 - Repository metadata, file tree, content access
  - Endpoints:
    - `/repos/{owner}/{repo}` - Repo metadata (description, language, stars)
    - `/repos/{owner}/{repo}/git/trees/{branch}` - Full file tree
    - `/repos/{owner}/{repo}/contents/{path}` - File content retrieval
  - Usage: `tensor/deepwiki_navigator.py`, `tensor/autonomous_training.py`
  - Client: requests library
  - Auth: `GITHUB_TOKEN` environment variable (optional, increases rate limit from 60 to 5000 req/hr)
  - Rate limit: 60 requests/hour (unauthenticated), 5000/hour (authenticated)
  - Rate limiting: Enforced in `tensor/bootstrap_manager.py` via `rate_limit_seconds` parameter

## Data Storage

**Databases:**
- SQLite (via Optuna) - Hyperparameter optimization studies
  - Location: `tensor/data/optuna_*.db`
  - Usage: `tensor/meta_optimizer.py` creates studies with TPESampler
  - Client: optuna library (handles SQLite internally)
  - Access: `optuna.load_study(storage=f"sqlite:///{path}")`

**File Storage:**
- Local filesystem only (no cloud storage)
  - Config/state: `tensor/data/` directory with JSON files
    - `fed_dates.json` - FOMC dates and NYSE holidays
    - `hdv_state.json` - HDV vectors for cross-domain discovery
    - `universals.json` - Universal patterns
    - `function_library.json` - 575+ function definitions with 53 DEQs
    - `ingestion_journal.json` - 92 papers journaled with metadata
    - `active_domains.json` - Domain capability maps
    - `capability_maps.json` - Bootstrap resource patterns
    - `curriculum_patterns.json` - Learning curriculum metadata
  - Ingested data: `tensor/data/ingested/` (359 papers via RSS)
  - DeepWiki cache: `tensor/data/deepwiki_cache/` (adaptive CSS selectors)
  - Visualization: `tensor/data/viz/` (Optuna plots)

**Caching:**
- None (no Redis, Memcached, etc.)
- File-based caching only: DeepWiki HTML cache, CSS selector persistence

## Authentication & Identity

**Auth Provider:**
- None (no OAuth2, JWT, SSO)
- Optional GitHub token for increased API rate limits
  - Token passed via `GITHUB_TOKEN` environment variable
  - Set in requests headers: `Authorization: token {GITHUB_TOKEN}`
  - Usage: `tensor/bootstrap_manager.py`, `tensor/deepwiki_navigator.py`

**CORS:**
- Configured in `platform/backend/main.py`
  - Allow origins: `http://localhost:5173` (Vite dev), `http://localhost:3000` (prod frontend)
  - Allow credentials: True
  - Allow methods: All
  - Allow headers: All

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Rollbar, or external error tracking)

**Logs:**
- Console output via standard Python logging (implicit in print statements)
- Docker health check: HTTP GET `/health` endpoint at `platform/backend/main.py:70`
- Application health: GET `/` root endpoint returns status JSON

**Visualization Output:**
- Optuna plots saved to `tensor/data/viz/` as HTML files
- Generated via `tensor/meta_optimizer.py:generate_optuna_plots()`

## CI/CD & Deployment

**Hosting:**
- Local development: uvicorn dev server on port 8000, Vite dev server on port 5173
- Docker: `docker-compose.yml` with single service `stability-engine`
  - Image: Built from `stability_engine/Dockerfile` (python:3.11-slim)
  - Port mapping: 8000:8000
  - Auto-restart: unless-stopped
  - Non-root user: appuser

**Deployment:**
- `docker-compose up` starts the stability engine container
- Frontend can be deployed separately as static files (Vite build output)

**CI Pipeline:**
- None detected (no GitHub Actions, GitLab CI, Jenkins configs)
- Testing: `pytest tests/ -q` (2261 passed, 4 skipped) and `pytest platform/backend/tests/ -q`
- No automated deployment pipeline

## Environment Configuration

**Required env vars:**
- `GITHUB_TOKEN` (optional) - GitHub API authentication for higher rate limits
- `PYTHONPATH` (set by start.sh) - Project root and ecemath paths
- `PYTHONUNBUFFERED` (Docker) - Unbuffered output for live logs

**Optional env vars:**
- `SCRAPLING_*` - If Scrapling is configured for specific sites (not yet codified)

**Secrets location:**
- Environment variables only (no .env file pattern enforced, but `.env*` typically excluded via `.gitignore`)
- No secrets management system (Vault, AWS Secrets Manager, etc.)

## Webhooks & Callbacks

**Incoming:**
- None (platform is read-only from external sources)

**Outgoing:**
- None (no webhooks to external services)

## Rate Limiting & Quotas

**GitHub API:**
- 60 requests/hour (unauthenticated)
- 5000 requests/hour (with GITHUB_TOKEN)
- Enforced in code via `rate_limit_seconds` parameter in `tensor/bootstrap_manager.py` (default=1.0s between requests)

**arXiv:**
- 3 seconds minimum delay between requests (hardcoded in `tensor/arxiv_pdf_parser.py`)

**DeepWiki:**
- rate_limit_seconds in `tensor/deepwiki_navigator.py` (configurable, default unspecified)

**Scrapling:**
- Rate limiting configurable in `tensor/scrapling_ingestion.py` (inherits from Scrapling SDK)

## Web Ingestion Architecture

**Ingestion Pipeline:**
1. `tensor/scrapling_ingestion.py:ScraplingFetcher` - Prefer Scrapling, fallback to requests
2. `tensor/financial_ingestion.py:FinancialIngestionRouter` - Domain-aware routing:
   - arXiv, EDGAR → `Fetcher` (basic HTTP)
   - Reuters, FT → `StealthyFetcher` (bypass anti-bot)
   - Seeking Alpha, MorningStar → `DynamicFetcher` (JS rendering required)
3. Article parsing: `tensor/semantic_flow_encoder.py` encodes articles to HDV vectors
4. Persistence: Parsed articles + metadata to JSON in `tensor/data/ingested/`

**Source Profiles:**
- Per-domain CSS selectors + fetcher choice + parser type
- Adaptive: `tensor/financial_ingestion.py:AdaptiveElementStore` learns and persists selectors

## Data Flow to APIs

**Outgoing Data:**
- `/api/v1/regime` - Classification results (POST payload → regime type + metrics)
- `/api/v1/calendar` - Calendar regime encoding (GET/POST → 5-channel amplitudes)
- `/api/v1/codegen` - Rust code generation (POST IntentSpec → compiled binary info)
- `/api/v1/hdv` - Cross-domain discovery (POST text/code → HDV vectors + similarity)
- `/api/v1/physics` - System simulation (POST parameters → eigenvalues + stability)
- `/api/v1/circuit` - Circuit optimization (POST target → Pareto solutions + frequency response)

**Incoming Data:**
- None (all APIs are outgoing/reactive; no external data push)

---

*Integration audit: 2026-02-22*

# Unified Tensor System Platform

Local-only intelligence platform with a FastAPI backend and React frontend.
No external API calls — all intelligence comes from local project models.

## Architecture

```
platform/
  backend/
    main.py                  FastAPI app (port 8000)
    routers/
      regime.py              GET /api/v1/regime/status
      calendar.py            GET /api/v1/calendar/phase, /range
      codegen.py             POST /api/v1/codegen/generate, GET /templates
      hdv.py                 POST /api/v1/hdv/encode, GET /universals
      physics.py             POST /api/v1/physics/simulate
    requirements.txt
    start.sh
    tests/
      test_api_routes.py     32 tests (pytest)
  frontend/
    src/
      App.jsx                5-tab navigation
      components/
        RegimeDashboard.jsx  LCA patch monitor (polls every 5s)
        CalendarOverlay.jsx  5-channel market phase + resonance detection
        CodeGenPanel.jsx     Domain/operation -> Rust code generation
        PhysicsSimulator.jsx RLC / Harmonic / Duffing + Koopman analysis
        HDVExplorer.jsx      HDV encoding + 2D PCA scatter
      api/client.js          Typed fetch wrappers for all endpoints
    package.json
    vite.config.js           Proxies /api to localhost:8000
    tailwind.config.js       Dark slate theme
    index.html
```

## Intelligence sources

| View | Local model |
|------|-------------|
| Regime Dashboard | `tensor.lca_patch_detector.LCAPatchDetector` |
| Calendar & Events | `tensor.calendar_regime.CalendarRegimeEncoder` |
| Code Generation | `codegen.pipeline.CodeGenPipeline` + BorrowVector classifier |
| Physics Simulator | `optimization.rlc_evaluator`, `optimization.duffing_evaluator` |
| HDV Explorer | `tensor.integrated_hdv.IntegratedHDVSystem` + `CrossDimensionalDiscovery` |

## Start backend

```bash
cd platform/backend && bash start.sh
```

Requires conda env `tensor` with fastapi + uvicorn installed:

```bash
conda run -n tensor pip install fastapi "uvicorn[standard]" httpx pydantic
```

## Start frontend (dev)

```bash
cd platform/frontend && npm install && npm run dev
```

Access at http://localhost:5173 (Vite proxies `/api` to port 8000).

## Run backend tests

```bash
# From project root
conda run -n tensor python -m pytest platform/backend/tests/ -q
```

## API overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/regime/status` | GET | LCA / nonabelian / chaotic classification |
| `/api/v1/calendar/phase?date=YYYY-MM-DD` | GET | 5-channel calendar phase vector |
| `/api/v1/calendar/range?start=...&end=...` | GET | Phase series over date range |
| `/api/v1/codegen/templates` | GET | All registered Rust templates |
| `/api/v1/codegen/generate` | POST | Generate Rust from domain + operation |
| `/api/v1/hdv/encode` | POST | Encode text to HDV + 2D PCA projection |
| `/api/v1/hdv/universals` | GET | Cross-domain universal patterns |
| `/api/v1/physics/simulate` | POST | RLC / harmonic / Duffing simulation |

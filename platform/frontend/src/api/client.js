/**
 * API client for the Unified Tensor System backend.
 * All requests are proxied through Vite to http://localhost:8000.
 */

const BASE = '/api/v1'

async function json(response) {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`HTTP ${response.status}: ${text}`)
  }
  return response.json()
}

export const api = {
  // Regime
  getRegimeStatus: () =>
    fetch(`${BASE}/regime/status`).then(json),

  // Calendar
  getCalendarPhase: (date) =>
    fetch(`${BASE}/calendar/phase${date ? `?date=${date}` : ''}`).then(json),

  getCalendarRange: (start, end) =>
    fetch(`${BASE}/calendar/range?start=${start}&end=${end}`).then(json),

  // CodeGen
  generateCode: (domain, operation, parameters = {}) =>
    fetch(`${BASE}/codegen/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain, operation, parameters }),
    }).then(json),

  getTemplates: () =>
    fetch(`${BASE}/codegen/templates`).then(json),

  // Physics
  simulate: (system_type, params = {}, n_steps = 400) =>
    fetch(`${BASE}/physics/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system_type, params, n_steps }),
    }).then(json),

  // HDV
  encodeHDV: (text, domain) =>
    fetch(`${BASE}/hdv/encode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, domain }),
    }).then(json),

  getUniversals: () =>
    fetch(`${BASE}/hdv/universals`).then(json),

  // Circuit Optimizer
  optimizeCircuit: (req) =>
    fetch('/api/v1/circuit/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }).then(json),
}

import { useState } from 'react'
import { Atom, Play, ChevronDown } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend,
} from 'recharts'
import { api } from '../api/client.js'

const SYSTEMS = {
  rlc: {
    label: 'RLC Circuit',
    description: 'L·i\' + R·i + q/C = 0  (KVL RK4)',
    params: [
      { key: 'R', label: 'R (Ω)', default: 10.0, step: 1, min: 0.1 },
      { key: 'L', label: 'L (H)', default: 0.01, step: 0.001, min: 1e-6 },
      { key: 'C', label: 'C (F)', default: 0.000001, step: 0.0000001, min: 1e-12 },
      { key: 'target_hz', label: 'Target (Hz)', default: 1000.0, step: 100, min: 1 },
    ],
  },
  harmonic: {
    label: 'Harmonic Oscillator',
    description: 'x\'\' + 2ζω₀x\' + ω₀²x = 0',
    params: [
      { key: 'omega0', label: 'ω₀ (rad/s)', default: 6.283, step: 0.5, min: 0.1 },
      { key: 'zeta', label: 'ζ (damping)', default: 0.1, step: 0.01, min: 0.001 },
      { key: 'x0', label: 'x₀', default: 1.0, step: 0.1 },
      { key: 'v0', label: 'v₀', default: 0.0, step: 0.1 },
    ],
  },
  duffing: {
    label: 'Duffing Oscillator',
    description: 'ẍ + δẋ + αx + βx³ = 0  (Koopman EDMD)',
    params: [
      { key: 'alpha', label: 'α (linear)', default: 1.0, step: 0.1, min: 0.01 },
      { key: 'beta', label: 'β (cubic)', default: 0.1, step: 0.05 },
      { key: 'delta', label: 'δ (damping)', default: 0.3, step: 0.05, min: 0 },
      { key: 'x0', label: 'x₀', default: 1.0, step: 0.1 },
      { key: 'v0', label: 'v₀', default: 0.0, step: 0.1 },
    ],
  },
}

const REGIME_STYLE = {
  lca:            { color: 'text-emerald-300', bg: 'bg-emerald-900/40 border-emerald-700' },
  overdamped:     { color: 'text-sky-300',     bg: 'bg-sky-900/40 border-sky-700' },
  abelian:        { color: 'text-emerald-300', bg: 'bg-emerald-900/40 border-emerald-700' },
  nonabelian:     { color: 'text-amber-300',   bg: 'bg-amber-900/40 border-amber-700' },
  near_separatrix:{ color: 'text-red-300',     bg: 'bg-red-900/40 border-red-700' },
}

function TrustGauge({ trust }) {
  const pct = Math.min(100, trust * 100)
  const color = trust > 0.6 ? '#10b981' : trust > 0.3 ? '#f59e0b' : '#ef4444'
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">Koopman trust</span>
        <span className="font-mono font-semibold" style={{ color }}>{trust.toFixed(4)}</span>
      </div>
      <div className="metric-bar">
        <div className="metric-bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">t = {Number(label).toFixed(3)}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.dataKey}: {Number(p.value).toFixed(5)}
        </p>
      ))}
    </div>
  )
}

export default function PhysicsSimulator() {
  const [systemType, setSystemType] = useState('duffing')
  const [paramValues, setParamValues] = useState(() => {
    const out = {}
    for (const [sys, cfg] of Object.entries(SYSTEMS)) {
      out[sys] = {}
      cfg.params.forEach(p => { out[sys][p.key] = p.default })
    }
    return out
  })
  const [nSteps, setNSteps] = useState(300)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const sysCfg = SYSTEMS[systemType]
  const currentParams = paramValues[systemType]

  const setParam = (key, val) => {
    setParamValues(prev => ({
      ...prev,
      [systemType]: { ...prev[systemType], [key]: val },
    }))
  }

  const handleSimulate = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.simulate(systemType, currentParams, nSteps)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Downsample trajectory to at most 300 points for chart perf
  const chartData = (() => {
    if (!result?.trajectory?.length) return []
    const traj = result.trajectory
    const stride = Math.max(1, Math.floor(traj.length / 300))
    return traj.filter((_, i) => i % stride === 0)
  })()

  const regimeStyle = REGIME_STYLE[result?.regime_type] || { color: 'text-slate-300', bg: 'bg-slate-700/30 border-slate-600' }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
          <Atom size={20} className="text-indigo-400" />
          Physics Simulator
        </h2>
        <p className="text-sm text-slate-400 mt-0.5">
          RK4 integration + Koopman EDMD analysis
        </p>
      </div>

      {/* System selector */}
      <div className="card">
        <div className="flex items-end gap-4 mb-4">
          <div className="flex-1">
            <label className="label">System Type</label>
            <div className="relative">
              <select
                value={systemType}
                onChange={e => { setSystemType(e.target.value); setResult(null) }}
                className="select pr-8"
              >
                {Object.entries(SYSTEMS).map(([k, v]) => (
                  <option key={k} value={k}>{v.label}</option>
                ))}
              </select>
              <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
            </div>
          </div>
          <div className="w-28">
            <label className="label">Steps</label>
            <input
              type="number"
              value={nSteps}
              onChange={e => setNSteps(Math.max(10, Math.min(600, parseInt(e.target.value) || 300)))}
              className="input"
              min={10} max={600}
            />
          </div>
        </div>

        <p className="text-xs text-slate-500 font-mono mb-4">{sysCfg.description}</p>

        {/* Param inputs */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          {sysCfg.params.map(p => (
            <div key={p.key}>
              <label className="label">{p.label}</label>
              <input
                type="number"
                value={currentParams[p.key]}
                step={p.step}
                min={p.min}
                onChange={e => setParam(p.key, parseFloat(e.target.value) || 0)}
                className="input font-mono"
              />
            </div>
          ))}
        </div>

        <button onClick={handleSimulate} className="btn-primary w-full" disabled={loading}>
          <Play size={14} />
          {loading ? 'Running...' : 'Run Simulation'}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-700 bg-red-900/20 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {result && (
        <>
          {/* Trajectory chart */}
          <div className="card">
            <div className="section-title">Trajectory x(t)</div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis
                  dataKey="t"
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  tickFormatter={v => v.toFixed(2)}
                  label={{ value: 't (s)', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 10 }}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  width={50}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                <Line
                  type="monotone"
                  dataKey="x"
                  stroke="#6366f1"
                  dot={false}
                  strokeWidth={1.5}
                  name="position x"
                />
                <Line
                  type="monotone"
                  dataKey="v"
                  stroke="#10b981"
                  dot={false}
                  strokeWidth={1.5}
                  name="velocity v"
                  strokeDasharray="4 2"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Koopman + regime summary */}
          <div className="grid grid-cols-3 gap-4">
            <div className="card text-center">
              <div className="text-xs text-slate-400 mb-1">ω₀</div>
              <div className="text-lg font-bold font-mono text-indigo-300">
                {result.omega0.toFixed(4)}
              </div>
              <div className="text-xs text-slate-500">rad/s</div>
            </div>
            <div className="card text-center">
              <div className="text-xs text-slate-400 mb-1">Q factor</div>
              <div className="text-lg font-bold font-mono text-sky-300">
                {result.Q.toFixed(4)}
              </div>
            </div>
            <div className={`card text-center border ${regimeStyle.bg}`}>
              <div className="text-xs text-slate-400 mb-1">Regime</div>
              <div className={`text-base font-bold uppercase ${regimeStyle.color}`}>
                {result.regime_type}
              </div>
            </div>
          </div>

          {/* Trust gauge */}
          <div className="card">
            <TrustGauge trust={result.koopman_trust} />
          </div>
        </>
      )}
    </div>
  )
}

import { useState, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  Legend,
  Cell,
} from 'recharts'
import { api } from '../api/client'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(v, decimals = 3) {
  if (v === undefined || v === null) return '—'
  return Number(v).toFixed(decimals)
}

function formatOhms(r) {
  if (r >= 1e6) return `${fmt(r / 1e6, 2)} MΩ`
  if (r >= 1e3) return `${fmt(r / 1e3, 2)} kΩ`
  return `${fmt(r, 2)} Ω`
}
function formatHenry(l) {
  if (l >= 1) return `${fmt(l, 3)} H`
  if (l >= 1e-3) return `${fmt(l * 1e3, 3)} mH`
  if (l >= 1e-6) return `${fmt(l * 1e6, 3)} μH`
  return `${fmt(l * 1e9, 3)} nH`
}
function formatFarad(c) {
  if (c >= 1e-3) return `${fmt(c * 1e3, 3)} mF`
  if (c >= 1e-6) return `${fmt(c * 1e6, 3)} μF`
  if (c >= 1e-9) return `${fmt(c * 1e9, 3)} nF`
  return `${fmt(c * 1e12, 3)} pF`
}

const REGIME_COLOR = {
  lca: '#34d399',        // green
  nonabelian: '#fbbf24', // amber
  chaotic: '#f87171',    // red
}
const REGIME_LABEL = {
  lca: 'LCA',
  nonabelian: 'NonAbelian',
  chaotic: 'Chaotic',
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SolutionCard({ sol, label, active, onClick }) {
  if (!sol) return null
  const regimeColor = REGIME_COLOR[sol.regime_type] || '#94a3b8'
  const regimeLabel = REGIME_LABEL[sol.regime_type] || sol.regime_type

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border transition-all ${
        active
          ? 'border-indigo-500 bg-indigo-950/60'
          : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-slate-300">{label}</span>
        <span
          className="text-xs px-1.5 py-0.5 rounded font-mono"
          style={{ background: regimeColor + '22', color: regimeColor }}
        >
          {regimeLabel}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-1 text-xs font-mono text-slate-300 mb-1.5">
        <span>R = {formatOhms(sol.R)}</span>
        <span>L = {formatHenry(sol.L)}</span>
        <span>C = {formatFarad(sol.C)}</span>
      </div>
      <div className="flex items-center gap-3 text-xs text-slate-400">
        <span>Q = {fmt(sol.Q_achieved, 2)}</span>
        <span>ω₀ = {fmt(sol.omega0_achieved, 0)} rad/s</span>
        <span>err = {fmt(sol.eigenvalue_error, 4)}</span>
      </div>
    </button>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CircuitOptimizer() {
  // Form state
  const [topology, setTopology] = useState('bandpass_rlc')
  const [centerFreqHz, setCenterFreqHz] = useState(1000)
  const [qTarget, setQTarget] = useState(5)
  const [maxPowerW, setMaxPowerW] = useState(1.0)

  // Result state
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('best_eigenvalue')

  const handleOptimize = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.optimizeCircuit({
        topology,
        center_freq_hz: Number(centerFreqHz),
        Q_target: Number(qTarget),
        max_power_w: Number(maxPowerW),
        component_tolerances: { R: 0.05, L: 0.10, C: 0.05 },
        weights: [1.0, 0.3, 0.5, 0.1],
      })
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [topology, centerFreqHz, qTarget, maxPowerW])

  // Prepare basin bar data
  const basinBars = result
    ? [
        { name: 'LCA', count: result.basin.n_lca, fill: '#34d399' },
        { name: 'NonAbelian', count: result.basin.n_nonabelian, fill: '#fbbf24' },
        { name: 'Chaotic', count: result.basin.n_chaotic, fill: '#f87171' },
      ]
    : []

  const activeSol = result?.pareto?.[activeTab]

  return (
    <div className="flex flex-col gap-6">
      {/* ── Top row: Left panel (spec) + Center panel (freq response) ── */}
      <div className="grid grid-cols-3 gap-4">
        {/* ── Left panel: Circuit Specification ── */}
        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 flex flex-col gap-4">
          <h2 className="text-sm font-semibold text-slate-200 tracking-wide">
            Circuit Specification
          </h2>

          {/* Topology */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400">Topology</label>
            <select
              value={topology}
              onChange={e => setTopology(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="bandpass_rlc">Bandpass RLC</option>
              <option value="lowpass_rc">Lowpass RC</option>
              <option value="highpass_rc">Highpass RC</option>
            </select>
          </div>

          {/* Center frequency */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400">Center Frequency (Hz)</label>
            <input
              type="number"
              min={1}
              step={100}
              value={centerFreqHz}
              onChange={e => setCenterFreqHz(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>

          {/* Q target slider */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400">
              Q Target: <span className="text-slate-200 font-mono">{qTarget}</span>
            </label>
            <input
              type="range"
              min={0.5}
              max={20}
              step={0.5}
              value={qTarget}
              onChange={e => setQTarget(Number(e.target.value))}
              className="accent-indigo-500"
            />
            <div className="flex justify-between text-xs text-slate-500">
              <span>0.5</span>
              <span>20</span>
            </div>
          </div>

          {/* Max power */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400">Max Power (W)</label>
            <input
              type="number"
              min={0.01}
              step={0.1}
              value={maxPowerW}
              onChange={e => setMaxPowerW(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>

          {/* Optimize button */}
          <button
            onClick={handleOptimize}
            disabled={loading}
            className="mt-auto bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-semibold rounded-lg py-2.5 transition-colors"
          >
            {loading ? 'Optimizing…' : 'OPTIMIZE'}
          </button>

          {error && (
            <p className="text-xs text-red-400 mt-1 break-all">{error}</p>
          )}
        </div>

        {/* ── Center panel: Frequency Response ── */}
        <div className="col-span-2 bg-slate-800/60 border border-slate-700 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-slate-200 tracking-wide">
              Frequency Response
            </h2>
            {result && (
              <span className="text-xs text-slate-500 font-mono">
                f₀ = {Number(result.target.center_freq_hz).toLocaleString()} Hz &nbsp;|&nbsp;
                Q = {fmt(result.target.Q_target, 1)}
              </span>
            )}
          </div>

          {result ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={result.frequency_response} margin={{ top: 4, right: 16, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="freq_hz"
                  scale="log"
                  domain={['dataMin', 'dataMax']}
                  type="number"
                  tickFormatter={v => {
                    if (v >= 1e6) return `${(v / 1e6).toFixed(0)}M`
                    if (v >= 1e3) return `${(v / 1e3).toFixed(0)}k`
                    return v.toFixed(0)
                  }}
                  label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  stroke="#475569"
                />
                <YAxis
                  yAxisId="mag"
                  dataKey="magnitude_db"
                  label={{ value: 'Magnitude (dB)', angle: -90, position: 'insideLeft', offset: 12, fill: '#64748b', fontSize: 11 }}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  stroke="#475569"
                />
                <YAxis
                  yAxisId="phase"
                  orientation="right"
                  domain={[-180, 180]}
                  label={{ value: 'Phase (°)', angle: 90, position: 'insideRight', offset: -10, fill: '#64748b', fontSize: 11 }}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  stroke="#475569"
                />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                  labelFormatter={v => `${Number(v).toLocaleString(undefined, { maximumFractionDigits: 1 })} Hz`}
                  formatter={(value, name) => [
                    name === 'magnitude_db' ? `${fmt(value, 2)} dB` : `${fmt(value, 1)}°`,
                    name === 'magnitude_db' ? 'Magnitude' : 'Phase',
                  ]}
                />
                <ReferenceLine yAxisId="mag" y={-3} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: '-3 dB', fill: '#f59e0b', fontSize: 10, position: 'right' }} />
                <Line
                  yAxisId="mag"
                  type="monotone"
                  dataKey="magnitude_db"
                  stroke="#818cf8"
                  dot={false}
                  strokeWidth={2}
                  name="magnitude_db"
                />
                <Line
                  yAxisId="phase"
                  type="monotone"
                  dataKey="phase_deg"
                  stroke="#34d399"
                  dot={false}
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  name="phase_deg"
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, color: '#94a3b8', paddingTop: 8 }}
                  formatter={name => name === 'magnitude_db' ? 'Magnitude (dB)' : 'Phase (°)'}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64 text-slate-600 text-sm">
              Press OPTIMIZE to compute frequency response
            </div>
          )}
        </div>
      </div>

      {/* ── Bottom row: Pareto Results + Stability Basin ── */}
      {result && (
        <div className="grid grid-cols-2 gap-4">
          {/* ── Right panel: Pareto Results ── */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
            <h2 className="text-sm font-semibold text-slate-200 tracking-wide mb-3">
              Pareto Solutions
            </h2>

            {/* Tab bar */}
            <div className="flex gap-1 mb-3">
              {[
                ['best_eigenvalue', 'Best Match'],
                ['best_stability', 'Best Stable'],
                ['best_cost', 'Best Cost'],
              ].map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                    activeTab === key
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Active solution detail */}
            {activeSol && (
              <div className="border border-slate-700 rounded-xl p-4 space-y-3">
                <div className="grid grid-cols-3 gap-2">
                  {[
                    ['R', formatOhms(activeSol.R)],
                    ['L', formatHenry(activeSol.L)],
                    ['C', formatFarad(activeSol.C)],
                  ].map(([lbl, val]) => (
                    <div key={lbl} className="bg-slate-700/50 rounded-lg p-2 text-center">
                      <div className="text-xs text-slate-500">{lbl}</div>
                      <div className="text-sm font-mono text-slate-200 mt-0.5">{val}</div>
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400 font-mono">
                  <span>Q achieved: <span className="text-slate-200">{fmt(activeSol.Q_achieved, 3)}</span></span>
                  <span>ω₀: <span className="text-slate-200">{fmt(activeSol.omega0_achieved, 1)} rad/s</span></span>
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400 font-mono">
                  <span>
                    Regime:{' '}
                    <span style={{ color: REGIME_COLOR[activeSol.regime_type] || '#94a3b8' }}>
                      {REGIME_LABEL[activeSol.regime_type] || activeSol.regime_type}
                    </span>
                    {' '}
                    {activeSol.converged
                      ? <span className="text-emerald-400">stable</span>
                      : <span className="text-red-400">not converged</span>}
                  </span>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                  Eig error: <span className="text-slate-200">{fmt(activeSol.eigenvalue_error, 4)}</span>
                  {'  '}Cost: <span className="text-slate-200">{fmt(activeSol.cost, 4)}</span>
                </div>
                {/* Target eigenvalues */}
                <div className="border-t border-slate-700 pt-2">
                  <div className="text-xs text-slate-500 mb-1">Target eigenvalues</div>
                  {result.target.target_eigenvalues.map(([re, im], i) => (
                    <div key={i} className="text-xs font-mono text-slate-400">
                      s{i + 1} = {fmt(re, 2)} {im >= 0 ? '+' : ''}{fmt(im, 2)}j
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ── Bottom panel: Stability Basin ── */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-sm font-semibold text-slate-200 tracking-wide">
                Stability Basin
              </h2>
              <span className="text-xs text-slate-500">
                worst-case error: <span className="text-slate-300 font-mono">{fmt(result.basin.worst_case_error, 4)}</span>
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              {result.basin.n_samples}-sample ±5% tolerance analysis
            </p>

            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={basinBars} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} stroke="#475569" />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} stroke="#475569" allowDecimals={false} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                  cursor={{ fill: '#334155' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {basinBars.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
              <div>
                <div className="text-emerald-400 font-semibold">{result.basin.n_lca}</div>
                <div className="text-slate-500">LCA ({fmt(result.basin.lca_fraction * 100, 0)}%)</div>
              </div>
              <div>
                <div className="text-amber-400 font-semibold">{result.basin.n_nonabelian}</div>
                <div className="text-slate-500">NonAbelian</div>
              </div>
              <div>
                <div className="text-red-400 font-semibold">{result.basin.n_chaotic}</div>
                <div className="text-slate-500">Chaotic</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

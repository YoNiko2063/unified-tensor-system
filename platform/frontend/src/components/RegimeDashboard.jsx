import { useState, useEffect, useCallback } from 'react'
import { Activity, RefreshCw, Cpu, Zap, AlertTriangle } from 'lucide-react'
import { api } from '../api/client.js'

const REGIME_CONFIG = {
  lca: {
    label: 'LCA',
    sublabel: 'Lie-Commutative Algebra',
    color: 'emerald',
    icon: Activity,
    description: 'Linear commutative operators — Koopman-navigable',
  },
  nonabelian: {
    label: 'NON-ABELIAN',
    sublabel: 'Koopman navigable',
    color: 'amber',
    icon: Zap,
    description: 'Non-commutative operators — requires curvature compensation',
  },
  chaotic: {
    label: 'CHAOTIC',
    sublabel: 'Spectrally intractable',
    color: 'red',
    icon: AlertTriangle,
    description: 'High curvature, low Koopman trust — avoid Lie-algebraic methods',
  },
}

const COLORS = {
  emerald: {
    card: 'border-emerald-600 bg-emerald-900/20',
    glow: 'shadow-emerald-500/20',
    dot: 'bg-emerald-400',
    text: 'text-emerald-300',
    bar: 'bg-emerald-500',
    badge: 'badge-green',
    ring: 'ring-emerald-500/40',
  },
  amber: {
    card: 'border-amber-600 bg-amber-900/20',
    glow: 'shadow-amber-500/20',
    dot: 'bg-amber-400',
    text: 'text-amber-300',
    bar: 'bg-amber-500',
    badge: 'badge-amber',
    ring: 'ring-amber-500/40',
  },
  red: {
    card: 'border-red-600 bg-red-900/20',
    glow: 'shadow-red-500/20',
    dot: 'bg-red-400',
    text: 'text-red-300',
    bar: 'bg-red-500',
    badge: 'badge-red',
    ring: 'ring-red-500/40',
  },
}

function MetricBar({ label, value, color = 'indigo', max = 1.0 }) {
  const pct = Math.min(100, (value / max) * 100)
  const barColors = {
    emerald: 'bg-emerald-500',
    amber: 'bg-amber-500',
    red: 'bg-red-500',
    indigo: 'bg-indigo-500',
    sky: 'bg-sky-500',
    violet: 'bg-violet-500',
  }
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-36 shrink-0">{label}</span>
      <div className="metric-bar flex-1">
        <div
          className={`metric-bar-fill ${barColors[color] || 'bg-indigo-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-slate-300 w-14 text-right tabular-nums">
        {value.toFixed(4)}
      </span>
    </div>
  )
}

function RegimeCard({ type, isActive }) {
  const cfg = REGIME_CONFIG[type] || REGIME_CONFIG.lca
  const c = COLORS[cfg.color]
  const Icon = cfg.icon

  return (
    <div
      className={`
        rounded-xl border p-4 flex flex-col gap-2 transition-all duration-300
        ${isActive ? `${c.card} ${c.glow} shadow-lg ring-1 ${c.ring}` : 'border-slate-700 bg-slate-800/50'}
      `}
    >
      <div className="flex items-center gap-2">
        <Icon size={14} className={isActive ? c.text : 'text-slate-500'} />
        <span className={`text-sm font-bold tracking-wide ${isActive ? c.text : 'text-slate-500'}`}>
          {cfg.label}
        </span>
        {isActive && (
          <span className="ml-auto flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${c.dot} animate-pulse`} />
            <span className={`text-xs font-semibold ${c.text}`}>ACTIVE</span>
          </span>
        )}
      </div>
      <p className={`text-xs leading-relaxed ${isActive ? 'text-slate-300' : 'text-slate-600'}`}>
        {cfg.sublabel}
      </p>
    </div>
  )
}

export default function RegimeDashboard() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.getRegimeStatus()
      setStatus(data)
      setError(null)
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  const patch = status?.patch_type || 'lca'
  const cfg = REGIME_CONFIG[patch] || REGIME_CONFIG.lca
  const c = COLORS[cfg.color]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Cpu size={20} className="text-indigo-400" />
            Regime Monitor
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            LCAPatchDetector on synthetic 2-D state — polls every 5s
          </p>
        </div>
        <button
          onClick={fetchStatus}
          className="btn-secondary text-xs"
          disabled={loading}
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-700 bg-red-900/20 px-4 py-3 text-sm text-red-300">
          Backend unreachable: {error}
        </div>
      )}

      {/* Regime cards */}
      <div className="grid grid-cols-3 gap-4">
        {['lca', 'nonabelian', 'chaotic'].map((type) => (
          <RegimeCard key={type} type={type} isActive={patch === type} />
        ))}
      </div>

      {/* Metrics panel */}
      <div className="card">
        <div className="section-title">Detector Metrics</div>
        {loading && !status ? (
          <div className="space-y-3 animate-pulse">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-4 bg-slate-700 rounded" />
            ))}
          </div>
        ) : status ? (
          <div className="space-y-3">
            <MetricBar
              label="Commutator norm"
              value={status.commutator_norm}
              color={status.commutator_norm > 0.3 ? 'red' : 'emerald'}
            />
            <MetricBar
              label="Curvature ratio"
              value={status.curvature_ratio}
              color={status.curvature_ratio > 0.5 ? 'amber' : 'sky'}
            />
            <MetricBar
              label="Koopman trust"
              value={status.koopman_trust}
              color={status.koopman_trust > 0.6 ? 'emerald' : status.koopman_trust > 0.3 ? 'amber' : 'red'}
            />
            <MetricBar
              label="Spectral gap"
              value={status.spectral_gap}
              color="violet"
            />
          </div>
        ) : null}
      </div>

      {/* Status footer */}
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>
          Current classification:{' '}
          <span className={`font-semibold ${c.text}`}>{patch.toUpperCase()}</span>
        </span>
        {lastUpdated && <span>Last updated: {lastUpdated}</span>}
      </div>

      {/* Description */}
      {status && (
        <div className={`rounded-lg border px-4 py-3 text-sm ${c.card}`}>
          <p className={c.text}>{cfg.description}</p>
        </div>
      )}
    </div>
  )
}

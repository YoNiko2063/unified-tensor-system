import { useState, useEffect } from 'react'
import { Network, Search, ChevronDown, Star } from 'lucide-react'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ZAxis,
} from 'recharts'
import { api } from '../api/client.js'

const VALID_DOMAINS = ['math', 'physics', 'behavioral', 'language', 'visual']

const DOMAIN_COLORS = {
  math:       '#6366f1',
  physics:    '#10b981',
  behavioral: '#f59e0b',
  language:   '#ec4899',
  visual:     '#8b5cf6',
}

const PRESETS = [
  { text: 'neural network gradient descent optimization', domain: 'math' },
  { text: 'RLC circuit resonance Koopman eigenvalue', domain: 'physics' },
  { text: 'code generation template borrow vector', domain: 'behavioral' },
  { text: 'Fourier transform frequency domain signal', domain: 'math' },
  { text: 'damped oscillator energy dissipation', domain: 'physics' },
  { text: 'reinforcement learning policy gradient', domain: 'behavioral' },
]

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs max-w-xs">
      <p className="font-semibold mb-1" style={{ color: DOMAIN_COLORS[d.domain] }}>{d.domain}</p>
      <p className="text-slate-300 truncate">{d.text}</p>
      <p className="text-slate-400 mt-1">norm: {d.norm?.toFixed(2)}</p>
    </div>
  )
}

export default function HDVExplorer() {
  const [text, setText] = useState('')
  const [domain, setDomain] = useState('physics')
  const [points, setPoints] = useState([])
  const [universals, setUniversals] = useState({ universals: [], count: 0, domains_active: [] })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load universals on mount
  useEffect(() => {
    api.getUniversals().then(setUniversals).catch(() => {})
  }, [])

  const handleEncode = async (t, d) => {
    const textToEncode = (t || text).trim()
    const domainToUse = d || domain
    if (!textToEncode) return

    setLoading(true)
    setError(null)
    try {
      const data = await api.encodeHDV(textToEncode, domainToUse)
      setPoints(prev => [
        ...prev.slice(-19),  // keep last 20
        {
          x: data.pca_2d[0],
          y: data.pca_2d[1],
          domain: data.domain,
          text: textToEncode,
          norm: data.norm,
          id: Date.now(),
        },
      ])
      // Refresh universals
      const u = await api.getUniversals()
      setUniversals(u)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handlePreset = (p) => {
    setText(p.text)
    setDomain(p.domain)
    handleEncode(p.text, p.domain)
  }

  // Group points by domain for scatter chart
  const domainGroups = {}
  for (const pt of points) {
    if (!domainGroups[pt.domain]) domainGroups[pt.domain] = []
    domainGroups[pt.domain].push(pt)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
          <Network size={20} className="text-indigo-400" />
          HDV Cross-Domain Explorer
        </h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Hyperdimensional vector encoding with 2D PCA projection — IntegratedHDVSystem
        </p>
      </div>

      {/* Encode input */}
      <div className="card">
        <div className="flex gap-3 mb-3">
          <div className="flex-1">
            <label className="label">Text to Encode</label>
            <input
              type="text"
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleEncode() }}
              className="input"
              placeholder="Enter concept, equation, or behavioral pattern..."
            />
          </div>
          <div className="w-36">
            <label className="label">Domain</label>
            <div className="relative">
              <select
                value={domain}
                onChange={e => setDomain(e.target.value)}
                className="select pr-8"
              >
                {VALID_DOMAINS.map(d => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
              <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
            </div>
          </div>
          <div className="flex items-end">
            <button onClick={() => handleEncode()} className="btn-primary" disabled={loading || !text.trim()}>
              <Search size={14} />
              {loading ? 'Encoding...' : 'Encode'}
            </button>
          </div>
        </div>

        {/* Preset buttons */}
        <div>
          <div className="section-title">Quick Presets</div>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p, i) => (
              <button
                key={i}
                onClick={() => handlePreset(p)}
                className="text-xs px-2.5 py-1 rounded-lg border border-slate-600 bg-slate-700/50
                           hover:border-indigo-500 hover:bg-slate-700 transition-colors text-slate-300"
              >
                <span className="font-mono text-slate-500">[{p.domain}]</span>{' '}
                {p.text.slice(0, 30)}{p.text.length > 30 ? '…' : ''}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-700 bg-red-900/20 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* 2D PCA scatter */}
      <div className="card">
        <div className="section-title">2D PCA Projection</div>

        {points.length === 0 ? (
          <div className="h-48 flex items-center justify-center text-slate-600 text-sm">
            Encode text above to see vectors projected here
          </div>
        ) : (
          <>
            {/* Domain legend */}
            <div className="flex flex-wrap gap-3 mb-3">
              {Object.keys(domainGroups).map(d => (
                <span key={d} className="flex items-center gap-1.5 text-xs text-slate-400">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: DOMAIN_COLORS[d] }}
                  />
                  {d} ({domainGroups[d].length})
                </span>
              ))}
            </div>

            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="PC1"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  label={{ value: 'PC 1', position: 'insideBottom', offset: -5, fill: '#475569', fontSize: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="PC2"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  width={40}
                  label={{ value: 'PC 2', angle: -90, position: 'insideLeft', fill: '#475569', fontSize: 10 }}
                />
                <ZAxis range={[40, 40]} />
                <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#334155' }} />
                {Object.entries(domainGroups).map(([d, pts]) => (
                  <Scatter
                    key={d}
                    name={d}
                    data={pts}
                    fill={DOMAIN_COLORS[d]}
                    fillOpacity={0.85}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </>
        )}
      </div>

      {/* Encoded vectors list */}
      {points.length > 0 && (
        <div className="card">
          <div className="section-title">Encoded Vectors ({points.length})</div>
          <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
            {[...points].reverse().map(pt => (
              <div key={pt.id} className="flex items-center gap-2 text-xs">
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: DOMAIN_COLORS[pt.domain] }}
                />
                <span className="text-slate-500 font-mono w-20 shrink-0">[{pt.domain}]</span>
                <span className="text-slate-300 truncate flex-1">{pt.text}</span>
                <span className="text-slate-500 font-mono tabular-nums shrink-0">
                  ({pt.x?.toFixed(2)}, {pt.y?.toFixed(2)})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Universals panel */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <div className="section-title mb-0">Cross-Domain Universals</div>
          <span className="badge-slate flex items-center gap-1">
            <Star size={10} className="text-amber-400" />
            {universals.count} found
          </span>
        </div>

        {universals.domains_active.length > 0 && (
          <div className="flex gap-2 mb-3 flex-wrap">
            {universals.domains_active.map(d => (
              <span key={d} className="text-xs px-2 py-0.5 rounded-full border border-slate-600 text-slate-400">
                {d}
              </span>
            ))}
          </div>
        )}

        {universals.count === 0 ? (
          <p className="text-xs text-slate-600">
            No universals yet — encode vectors from multiple domains to discover cross-domain patterns.
          </p>
        ) : (
          <div className="space-y-2">
            {universals.universals.map((u, i) => (
              <div key={i} className="flex items-start gap-3 p-2 rounded bg-slate-700/40 border border-slate-700">
                <Star size={12} className="text-amber-400 mt-0.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex gap-2 flex-wrap">
                    {u.domains.map(d => (
                      <span key={d} className="text-xs font-mono" style={{ color: DOMAIN_COLORS[d] }}>{d}</span>
                    ))}
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5 truncate">{u.pattern || `dim ${u.dimension}`}</p>
                </div>
                <span className="text-xs font-mono text-slate-400 tabular-nums shrink-0">
                  {u.confidence.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

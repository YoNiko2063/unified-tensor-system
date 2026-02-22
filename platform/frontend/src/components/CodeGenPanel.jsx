import { useState, useEffect } from 'react'
import { Code2, Zap, CheckCircle, XCircle, ChevronDown } from 'lucide-react'
import { api } from '../api/client.js'

const D_SEP = 0.43

function BorrowBar({ eBorrow }) {
  const pct = Math.min(100, (eBorrow / 0.6) * 100)
  const safe = eBorrow < D_SEP
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">E_borrow</span>
        <span className={`font-mono font-semibold ${safe ? 'text-emerald-400' : 'text-red-400'}`}>
          {eBorrow.toFixed(4)}
        </span>
      </div>
      <div className="metric-bar relative">
        <div
          className={`metric-bar-fill ${safe ? 'bg-emerald-500' : 'bg-red-500'}`}
          style={{ width: `${pct}%` }}
        />
        {/* D_SEP marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-amber-400"
          style={{ left: `${(D_SEP / 0.6) * 100}%` }}
          title={`D_sep = ${D_SEP}`}
        />
      </div>
      <div className="flex justify-between text-xs mt-0.5 text-slate-600">
        <span>0</span>
        <span className="text-amber-500">D_sep={D_SEP}</span>
        <span>0.6</span>
      </div>
    </div>
  )
}

function BorrowVector({ bv }) {
  const labels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
  const weights = [0.25, 0.18, 0.15, 0.17, 0.15, 0.10]
  return (
    <div>
      <div className="section-title">Borrow Vector</div>
      <div className="grid grid-cols-6 gap-1">
        {bv.map((v, i) => (
          <div key={i} className="flex flex-col items-center gap-1">
            <div className="w-full h-16 bg-slate-700 rounded relative overflow-hidden flex items-end">
              <div
                className="w-full bg-indigo-500/80 transition-all duration-500"
                style={{ height: `${Math.min(100, v * 100)}%` }}
              />
            </div>
            <span className="text-xs font-mono text-slate-400">{labels[i]}</span>
            <span className="text-xs font-mono text-slate-300 tabular-nums">{v.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <div className="text-xs text-slate-500 mt-1">
        Weights: [{weights.map(w => w.toFixed(2)).join(', ')}]
      </div>
    </div>
  )
}

export default function CodeGenPanel() {
  const [templates, setTemplates] = useState([])
  const [domain, setDomain] = useState('numeric')
  const [operation, setOperation] = useState('')
  const [paramStr, setParamStr] = useState('{}')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.getTemplates().then(data => {
      setTemplates(data)
      if (data.length > 0) {
        setDomain(data[0].domain)
        setOperation(data[0].operation)
      }
    }).catch(e => setError(e.message))
  }, [])

  // Get unique domains
  const domains = [...new Set(templates.map(t => t.domain))]
  // Ops for current domain
  const ops = templates.filter(t => t.domain === domain).map(t => t.operation)

  const handleDomainChange = (e) => {
    const d = e.target.value
    setDomain(d)
    const firstOp = templates.find(t => t.domain === d)?.operation || ''
    setOperation(firstOp)
    setResult(null)
  }

  const handleGenerate = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      let params = {}
      try { params = JSON.parse(paramStr) } catch { /* ignore */ }
      const data = await api.generateCode(domain, operation, params)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const currentTpl = templates.find(t => t.domain === domain && t.operation === operation)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
          <Code2 size={20} className="text-indigo-400" />
          Code Generation
        </h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Intent → BorrowVector classifier → Rust template
        </p>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="label">Domain</label>
            <div className="relative">
              <select
                value={domain}
                onChange={handleDomainChange}
                className="select pr-8"
              >
                {domains.map(d => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
              <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
            </div>
          </div>
          <div>
            <label className="label">Operation</label>
            <div className="relative">
              <select
                value={operation}
                onChange={e => { setOperation(e.target.value); setResult(null) }}
                className="select pr-8"
              >
                {ops.map(op => (
                  <option key={op} value={op}>{op}</option>
                ))}
              </select>
              <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
            </div>
          </div>
        </div>

        <div className="mb-4">
          <label className="label">Parameters (JSON)</label>
          <input
            type="text"
            value={paramStr}
            onChange={e => setParamStr(e.target.value)}
            className="input font-mono text-xs"
            placeholder='{"window": 20, "dtype": "f64"}'
          />
        </div>

        {currentTpl && (
          <p className="text-xs text-slate-500 mb-3 font-mono">
            {currentTpl.borrow_profile} — {currentTpl.description}
          </p>
        )}

        <button onClick={handleGenerate} className="btn-primary w-full" disabled={loading || !operation}>
          <Zap size={14} />
          {loading ? 'Generating...' : 'Generate Rust'}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-700 bg-red-900/20 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {result && (
        <>
          {/* Status row */}
          <div className="flex items-center gap-3 flex-wrap">
            {result.predicted_compiles ? (
              <span className="flex items-center gap-1.5 badge-green">
                <CheckCircle size={12} /> COMPILES
              </span>
            ) : (
              <span className="flex items-center gap-1.5 badge-red">
                <XCircle size={12} /> FAILS
              </span>
            )}
            <span className="badge-slate">
              template: {result.template_name}
            </span>
            <span className="badge-slate font-mono">
              P(ok)={result.probability.toFixed(4)}
            </span>
            {!result.success && (
              <span className="badge-amber">compile error detected</span>
            )}
          </div>

          {/* E_borrow bar */}
          <div className="card">
            <BorrowBar eBorrow={result.e_borrow} />
          </div>

          {/* Borrow vector bars */}
          <div className="card">
            <BorrowVector bv={result.borrow_vector} />
          </div>

          {/* Rust source */}
          <div className="card">
            <div className="section-title">Rust Source</div>
            <pre className="code-block whitespace-pre-wrap">
              {result.rust_source}
            </pre>
          </div>

          {/* Compile error snippet */}
          {result.error && (
            <div className="card border-red-700/50">
              <div className="section-title text-red-400">Compile Error (excerpt)</div>
              <pre className="font-mono text-xs text-red-300 whitespace-pre-wrap leading-relaxed">
                {result.error}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  )
}

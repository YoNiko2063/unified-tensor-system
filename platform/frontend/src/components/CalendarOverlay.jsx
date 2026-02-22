import { useState, useEffect } from 'react'
import { Calendar, AlertTriangle, TrendingUp, RefreshCw } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { api } from '../api/client.js'

const CHANNEL_COLORS = {
  earnings:  { bar: '#6366f1', label: 'text-indigo-300', bg: 'bg-indigo-900/30 border-indigo-700' },
  fed:       { bar: '#f59e0b', label: 'text-amber-300',  bg: 'bg-amber-900/30 border-amber-700' },
  options:   { bar: '#10b981', label: 'text-emerald-300', bg: 'bg-emerald-900/30 border-emerald-700' },
  rebalance: { bar: '#8b5cf6', label: 'text-violet-300', bg: 'bg-violet-900/30 border-violet-700' },
  holiday:   { bar: '#64748b', label: 'text-slate-300',  bg: 'bg-slate-700/30 border-slate-600' },
}

function today() {
  return new Date().toISOString().split('T')[0]
}

function PhaseBar({ channel, amplitude, isDominant }) {
  const cfg = CHANNEL_COLORS[channel] || CHANNEL_COLORS.holiday
  const pct = Math.min(100, amplitude * 100)

  return (
    <div className={`flex items-center gap-3 rounded-lg px-3 py-2 border ${isDominant ? cfg.bg : 'bg-transparent border-transparent'}`}>
      <span className={`text-xs font-mono w-20 shrink-0 ${cfg.label}`}>{channel}</span>
      <div className="flex-1 h-2 rounded-full bg-slate-700 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: cfg.bar }}
        />
      </div>
      <span className="text-xs font-mono text-slate-300 w-10 text-right tabular-nums">
        {amplitude.toFixed(2)}
      </span>
      {isDominant && (
        <span className="text-xs font-semibold text-indigo-400 w-16">DOMINANT</span>
      )}
    </div>
  )
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const { channel, amplitude } = payload[0].payload
  const cfg = CHANNEL_COLORS[channel] || {}
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg p-2 text-xs">
      <p className={`font-semibold ${cfg.label}`}>{channel}</p>
      <p className="text-slate-300">Amplitude: {amplitude.toFixed(4)}</p>
    </div>
  )
}

export default function CalendarOverlay() {
  const [date, setDate] = useState(today())
  const [phase, setPhase] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchPhase = async (d) => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getCalendarPhase(d)
      setPhase(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchPhase(date) }, [])

  const handleDateChange = (e) => {
    setDate(e.target.value)
  }

  const handleFetch = () => fetchPhase(date)

  const chartData = phase
    ? phase.channels.map((ch, i) => ({ channel: ch, amplitude: phase.amplitudes[i] }))
    : []

  const dominantIdx = phase
    ? phase.amplitudes.indexOf(Math.max(...phase.amplitudes))
    : -1

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Calendar size={20} className="text-indigo-400" />
            Calendar Regime
          </h2>
          <p className="text-sm text-slate-400 mt-0.5">
            5-channel market phase vector via CalendarRegimeEncoder
          </p>
        </div>
      </div>

      {/* Date picker */}
      <div className="card">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="label">Query Date</label>
            <input
              type="date"
              value={date}
              onChange={handleDateChange}
              className="input"
            />
          </div>
          <button onClick={handleFetch} className="btn-primary" disabled={loading}>
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            {loading ? 'Loading...' : 'Fetch Phase'}
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-700 bg-red-900/20 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {phase && (
        <>
          {/* Regime label + date */}
          <div className="flex items-center gap-3">
            <span className="text-slate-400 text-sm">{phase.date}</span>
            <span className="badge bg-indigo-900/60 text-indigo-300 border border-indigo-700">
              {phase.regime_label}
            </span>
            {phase.active_events.length > 0 && phase.active_events.map(ev => (
              <span key={ev} className="badge bg-slate-700/60 text-slate-300 border border-slate-600">
                {ev.replace('_', ' ')}
              </span>
            ))}
          </div>

          {/* Resonance warning */}
          {phase.resonance_detected && (
            <div className="flex items-start gap-3 rounded-lg border border-amber-700 bg-amber-900/20 px-4 py-3">
              <AlertTriangle size={16} className="text-amber-400 mt-0.5 shrink-0" />
              <div>
                <p className="text-sm font-semibold text-amber-300">
                  Calendar Resonance Detected
                </p>
                <p className="text-xs text-amber-400/80 mt-0.5">
                  Multiple cycles in 2:1 ratio — vol multiplier{' '}
                  <strong>{phase.vol_multiplier.toFixed(2)}x</strong>
                </p>
              </div>
            </div>
          )}

          {/* Phase bars */}
          <div className="card">
            <div className="section-title">Phase Vector</div>
            <div className="space-y-2">
              {phase.channels.map((ch, i) => (
                <PhaseBar
                  key={ch}
                  channel={ch}
                  amplitude={phase.amplitudes[i]}
                  isDominant={i === dominantIdx}
                />
              ))}
            </div>
          </div>

          {/* Bar chart */}
          <div className="card">
            <div className="section-title">Amplitude Chart</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={chartData} barSize={32}>
                <XAxis
                  dataKey="channel"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  domain={[0, 'dataMax + 0.05']}
                  width={45}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="amplitude" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.channel}
                      fill={CHANNEL_COLORS[entry.channel]?.bar || '#64748b'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Summary row */}
          <div className="grid grid-cols-3 gap-4">
            <div className="card text-center">
              <div className="text-xs text-slate-400 mb-1">Dominant Cycle</div>
              <div className="text-lg font-bold text-indigo-300 capitalize">{phase.dominant_cycle}</div>
            </div>
            <div className="card text-center">
              <div className="text-xs text-slate-400 mb-1">Vol Multiplier</div>
              <div className={`text-lg font-bold ${phase.resonance_detected ? 'text-amber-300' : 'text-emerald-300'}`}>
                {phase.vol_multiplier.toFixed(3)}x
              </div>
            </div>
            <div className="card text-center">
              <div className="text-xs text-slate-400 mb-1">Resonance</div>
              <div className={`text-lg font-bold ${phase.resonance_detected ? 'text-amber-300' : 'text-slate-400'}`}>
                {phase.resonance_detected ? 'YES' : 'NO'}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

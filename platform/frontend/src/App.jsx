import { useState } from 'react'
import { Activity, Calendar, Code2, Atom, Network, Terminal, Cpu } from 'lucide-react'
import RegimeDashboard from './components/RegimeDashboard.jsx'
import CalendarOverlay from './components/CalendarOverlay.jsx'
import CodeGenPanel from './components/CodeGenPanel.jsx'
import PhysicsSimulator from './components/PhysicsSimulator.jsx'
import HDVExplorer from './components/HDVExplorer.jsx'
import CircuitOptimizer from './components/CircuitOptimizer.jsx'

const TABS = [
  {
    id: 'regime',
    label: 'Regime',
    icon: Activity,
    description: 'LCA patch classification',
    component: RegimeDashboard,
  },
  {
    id: 'calendar',
    label: 'Calendar',
    icon: Calendar,
    description: '5-channel market phase',
    component: CalendarOverlay,
  },
  {
    id: 'codegen',
    label: 'Code Gen',
    icon: Code2,
    description: 'Intent → Rust pipeline',
    component: CodeGenPanel,
  },
  {
    id: 'physics',
    label: 'Physics',
    icon: Atom,
    description: 'RLC / Duffing simulator',
    component: PhysicsSimulator,
  },
  {
    id: 'hdv',
    label: 'HDV Explorer',
    icon: Network,
    description: 'Cross-domain discovery',
    component: HDVExplorer,
  },
  {
    id: 'circuit',
    label: 'Circuit Optimizer',
    icon: Cpu,
    description: 'Koopman-guided RLC optimization',
    component: CircuitOptimizer,
  },
]

function TabBar({ active, onSelect }) {
  return (
    <nav className="flex items-center gap-1 px-1">
      {TABS.map(tab => {
        const Icon = tab.icon
        const isActive = active === tab.id
        return (
          <button
            key={tab.id}
            onClick={() => onSelect(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-150
              ${isActive
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
              }
            `}
          >
            <Icon size={15} />
            <span>{tab.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('regime')
  const activeTabCfg = TABS.find(t => t.id === activeTab)
  const ActiveComponent = activeTabCfg?.component

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      {/* Top bar */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex items-center gap-6 h-14">
            {/* Logo */}
            <div className="flex items-center gap-2.5 shrink-0">
              <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
                <Terminal size={14} className="text-white" />
              </div>
              <div>
                <div className="text-sm font-bold text-slate-100 leading-none">Unified Tensor</div>
                <div className="text-xs text-slate-500 leading-none mt-0.5">local-only intelligence</div>
              </div>
            </div>

            {/* Tab bar */}
            <div className="flex-1">
              <TabBar active={activeTab} onSelect={setActiveTab} />
            </div>

            {/* Status indicator */}
            <div className="flex items-center gap-2 text-xs text-slate-500 shrink-0">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              localhost:8000
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Page title */}
        <div className="mb-6">
          <div className="flex items-center gap-3">
            {activeTabCfg && (
              <>
                <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center">
                  {activeTabCfg.icon && <activeTabCfg.icon size={16} className="text-indigo-400" />}
                </div>
                <div>
                  <h1 className="text-lg font-semibold text-slate-100">{activeTabCfg.label}</h1>
                  <p className="text-xs text-slate-500">{activeTabCfg.description}</p>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Tab content */}
        <div>
          {ActiveComponent && <ActiveComponent />}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-16 py-4">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between text-xs text-slate-600">
          <span>Unified Tensor System — no external API calls</span>
          <span className="font-mono">
            v1.0 · FastAPI + React 18 + Recharts
          </span>
        </div>
      </footer>
    </div>
  )
}

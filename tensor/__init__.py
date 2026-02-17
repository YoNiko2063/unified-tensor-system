"""Unified tensor framework T ∈ R^(L x N x N x t)."""
import importlib

_IMPORTS = [
    ('tensor.core',             'UnifiedTensor'),
    ('tensor.observer',         'TensorObserver'),
    ('tensor.code_graph',       'CodeGraph'),
    ('tensor.hardware_profiler', 'HardwareProfiler'),
    ('tensor.gsd_bridge',       'GSDBridge'),
    ('tensor.bootstrap',        'BootstrapOrchestrator'),
    ('tensor.explorer',         'ConfigurationExplorer'),
    ('tensor.market_graph',     'MarketGraph'),
    ('tensor.dev_agent_bridge', 'DevAgentBridge'),
    ('tensor.trading_bridge',   'TradingBridge'),
    ('tensor.code_validator',   'CodeValidator'),
    ('tensor.skill_writer',     'SkillWriter'),
    ('tensor.math_connections', 'MathConnections'),
    ('tensor.realtime_feed',    'RealtimeFeed'),
    ('tensor.neural_bridge',    'NeuralBridge'),
    ('tensor.scraper_bridge',   'ScraperBridge'),
    ('tensor.trajectory',       'LearningTrajectory'),
    ('tensor.agent_network',    'AgentNetwork'),
    ('tensor.domain_fibers',    'FiberBundle'),
]

for _mod, _cls in _IMPORTS:
    try:
        _m = importlib.import_module(_mod)
        globals()[_cls] = getattr(_m, _cls)
    except (ImportError, AttributeError):
        pass

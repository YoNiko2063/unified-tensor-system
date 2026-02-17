# Roadmap: Tensor-Guided Improvement

## Overview

5 phases, each targeting one high-tension module.
Guided by tensor L2 free energy minimization.

## Phases

- [ ] **Phase 1: Improve autoimprover-repo.src.dev_agent.cli.router** - Reduce free energy from -2.6316
- [ ] **Phase 2: Improve autoimprover-repo.src.dev_agent.ui.simple_tui** - Reduce free energy from -2.6220
- [ ] **Phase 3: Improve src.dev_agent.run_self_threads_step** - Reduce free energy from -2.6098
- [ ] **Phase 4: Improve autoimprover-repo.src.dev_agent.run_self_threads_step** - Reduce free energy from -2.6098
- [ ] **Phase 5: Improve autoimprover-repo.src.dev_agent.introspection.emit_system_snapshot** - Reduce free energy from -2.5959

## Phase Details

### Phase 1: Improve autoimprover-repo.src.dev_agent.cli.router
**Goal**: Reduce structural tension in autoimprover-repo.src.dev_agent.cli.router
**Depends on**: Nothing (first phase)
**Requirements**: REQ-01
**Success Criteria** (what must be TRUE):
  1. Free energy of autoimprover-repo.src.dev_agent.cli.router decreases by >20%
  2. Overall L2 consonance does not degrade
  3. All existing tests still pass

### Phase 2: Improve autoimprover-repo.src.dev_agent.ui.simple_tui
**Goal**: Reduce structural tension in autoimprover-repo.src.dev_agent.ui.simple_tui
**Depends on**: Phase 1
**Requirements**: REQ-02
**Success Criteria** (what must be TRUE):
  1. Free energy of autoimprover-repo.src.dev_agent.ui.simple_tui decreases by >20%
  2. Overall L2 consonance does not degrade
  3. All existing tests still pass

### Phase 3: Improve src.dev_agent.run_self_threads_step
**Goal**: Reduce structural tension in src.dev_agent.run_self_threads_step
**Depends on**: Phase 2
**Requirements**: REQ-03
**Success Criteria** (what must be TRUE):
  1. Free energy of src.dev_agent.run_self_threads_step decreases by >20%
  2. Overall L2 consonance does not degrade
  3. All existing tests still pass

### Phase 4: Improve autoimprover-repo.src.dev_agent.run_self_threads_step
**Goal**: Reduce structural tension in autoimprover-repo.src.dev_agent.run_self_threads_step
**Depends on**: Phase 3
**Requirements**: REQ-04
**Success Criteria** (what must be TRUE):
  1. Free energy of autoimprover-repo.src.dev_agent.run_self_threads_step decreases by >20%
  2. Overall L2 consonance does not degrade
  3. All existing tests still pass

### Phase 5: Improve autoimprover-repo.src.dev_agent.introspection.emit_system_snapshot
**Goal**: Reduce structural tension in autoimprover-repo.src.dev_agent.introspection.emit_system_snapshot
**Depends on**: Phase 4
**Requirements**: REQ-05
**Success Criteria** (what must be TRUE):
  1. Free energy of autoimprover-repo.src.dev_agent.introspection.emit_system_snapshot decreases by >20%
  2. Overall L2 consonance does not degrade
  3. All existing tests still pass

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|----------|
| 1. autoimprover-repo.src.dev_agent.cli.router | 0/1 | Not started | - |
| 2. autoimprover-repo.src.dev_agent.ui.simple_tui | 0/1 | Not started | - |
| 3. src.dev_agent.run_self_threads_step | 0/1 | Not started | - |
| 4. autoimprover-repo.src.dev_agent.run_self_threads_step | 0/1 | Not started | - |
| 5. autoimprover-repo.src.dev_agent.introspection.emit_system_snapshot | 0/1 | Not started | - |

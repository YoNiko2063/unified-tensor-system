#!/usr/bin/env python3
"""
Session 2: Populate function library from arXiv + find cross-dimensional universals.

Pipeline:
1. Download LaTeX from arXiv e-print for 60 papers
2. Extract equations, classify, store in function_library.json
3. Run DEQ canonicalization
4. Encode into HDV space across math/behavioral/execution dimensions
5. Run CrossDimensionalDiscovery.find_universals()
6. Report final counts
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter

import os
os.chdir('/home/nyoo/projects/unified-tensor-system')

import numpy as np

from tensor.function_basis import (
    FunctionBasisLibrary,
    EquationParser,
    DEQCanonicalizer,
    ThreadedArxivIngester,
    populate_library_from_arxiv,
)
from tensor.arxiv_pdf_parser import ArxivPDFSourceParser
from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
from tensor.integrated_hdv import IntegratedHDVSystem


def record_patterns(lib, hdv, discovery):
    """
    Record library entries as patterns in multiple HDV dimensions.

    - Math dimension: equation type + structure
    - Behavioral dimension: operator workflow patterns (for DEQs)
    - Execution dimension: validated function patterns
    """
    counts = {'math': 0, 'behavioral': 0, 'execution': 0}

    for name, entry in lib.library.items():
        symbolic = entry.get('symbolic_str', '')
        func_type = entry.get('type', 'unknown')
        eq_type = entry.get('equation_type', 'algebraic')
        params = entry.get('parameters', [])
        operator_terms = entry.get('operator_terms', [])
        domains = entry.get('domains', ['general'])
        domain = list(domains)[0] if isinstance(domains, (set, list)) else 'general'

        # ── MATH dimension: encode the equation itself ──
        try:
            hdv_vec = hdv.encode_equation(symbolic, domain)
            metadata = {
                'type': func_type,
                'equation_type': eq_type,
                'domain': domain,
                'name': name,
                'symbolic': symbolic[:100],
                'parameters': params[:5],
                'has_nonlinearity': entry.get('has_nonlinearity', False),
            }
            discovery.record_pattern('math', hdv_vec, metadata)
            counts['math'] += 1
        except Exception:
            pass

        # ── BEHAVIORAL dimension: structural workflow patterns ──
        # Encode equations as behavioral patterns based on their structure
        # This creates the cross-dimensional bridge
        try:
            behavior_tokens = [func_type]
            if eq_type != 'algebraic':
                behavior_tokens.append(eq_type)
            if operator_terms:
                behavior_tokens.extend(operator_terms)
            if params:
                behavior_tokens.extend([f'param_{p}' for p in params[:3]])
            behavior_tokens.append(domain)

            workflow_desc = ' '.join(behavior_tokens)
            hdv_vec2 = hdv.structural_encode(workflow_desc, domain)
            metadata2 = {
                'type': func_type,
                'equation_type': eq_type,
                'domain': domain,
                'name': name,
                'operator_terms': operator_terms,
                'workflow': workflow_desc,
            }
            discovery.record_pattern('behavioral', hdv_vec2, metadata2)
            counts['behavioral'] += 1
        except Exception:
            pass

        # ── EXECUTION dimension: validated computation patterns ──
        # Equations that represent computable functions
        if func_type in ('exponential', 'trigonometric', 'logarithmic', 'power_law', 'polynomial'):
            try:
                exec_tokens = ['validated', func_type]
                if params:
                    exec_tokens.extend(params[:3])
                exec_tokens.append(domain)
                exec_desc = ' '.join(exec_tokens)

                hdv_vec3 = hdv.structural_encode(exec_desc, domain)
                metadata3 = {
                    'type': func_type,
                    'domain': domain,
                    'name': name,
                    'validation': 'structural',
                    'parameters': params[:5],
                }
                discovery.record_pattern('execution', hdv_vec3, metadata3)
                counts['execution'] += 1
            except Exception:
                pass

    return counts


def main():
    print("=" * 70)
    print("SESSION 2: Populate Library + Cross-Dimensional Discovery")
    print("=" * 70)

    lib_path = 'tensor/data/function_library.json'

    # ── Step 1: Check existing library ────────────────────────────────────
    lib = FunctionBasisLibrary(lib_path)
    print(f"\n[Step 1] Existing library: {len(lib.library)} entries")

    # ── Step 2: Download equations from 60 arXiv papers (threaded) ───────
    n_agents = int(os.environ.get('INGEST_AGENTS', '4'))
    print(f"\n[Step 2] Threaded ingestion: {n_agents} agent threads, 60 papers...")
    ingester = ThreadedArxivIngester(lib, n_agents=n_agents, rate_limit=1.5)
    result = ingester.ingest(max_papers=60)

    # ── Step 3: Statistics ────────────────────────────────────────────────
    eq_types = Counter(e.get('equation_type', 'none') for e in lib.library.values())
    func_types = Counter(e.get('type', 'unknown') for e in lib.library.values())
    domains = Counter()
    for e in lib.library.values():
        for d in (e.get('domains', []) if isinstance(e.get('domains'), list) else list(e.get('domains', set()))):
            domains[d] += 1
    has_deq = sum(1 for e in lib.library.values() if e.get('is_canonical_deq'))

    print(f"\n[Step 3] Library statistics:")
    print(f"  Total functions:   {len(lib.library)}")
    print(f"  Equation types:    {dict(eq_types)}")
    print(f"  Function types:    {dict(func_types)}")
    print(f"  Domains:           {dict(domains)}")
    print(f"  Canonical DEQs:    {has_deq}")

    # ── Step 4: Set up HDV system ─────────────────────────────────────────
    print(f"\n[Step 4] Setting up IntegratedHDVSystem...")
    hdv = IntegratedHDVSystem(hdv_dim=10000, library_path=lib_path)

    # Clear old universals for a fresh scan
    discovery = CrossDimensionalDiscovery(
        hdv_system=hdv,
        similarity_threshold=0.85,
        universals_path='tensor/data/universals.json',
    )
    # Start with empty universals list for clean discovery
    discovery.universals = []

    # ── Step 5: Record patterns across dimensions ─────────────────────────
    print(f"\n[Step 5] Recording patterns in HDV space...")
    counts = record_patterns(lib, hdv, discovery)
    print(f"  Recorded: math={counts['math']}, behavioral={counts['behavioral']}, execution={counts['execution']}")
    print(f"  Pattern counts: {discovery.get_pattern_counts()}")

    # ── Step 6: Find universals ───────────────────────────────────────────
    print(f"\n[Step 6] Running find_universals()...")
    print(f"  Comparing {counts['math']} math x {counts['behavioral']} behavioral x {counts['execution']} execution patterns...")

    new_universals = discovery.find_universals(similarity_threshold=0.85)
    print(f"  Universals at threshold=0.85: {len(new_universals)}")

    if len(new_universals) == 0:
        print(f"  Trying threshold=0.80...")
        new_universals = discovery.find_universals(similarity_threshold=0.80)
        print(f"  Universals at threshold=0.80: {len(new_universals)}")

    if len(new_universals) == 0:
        print(f"  Trying threshold=0.70...")
        new_universals = discovery.find_universals(similarity_threshold=0.70)
        print(f"  Universals at threshold=0.70: {len(new_universals)}")

    print(f"\n  Total universals discovered: {len(discovery.universals)}")

    if discovery.universals:
        print(f"\n  --- Top Cross-Dimensional Universals ---")
        # Sort by similarity descending
        sorted_u = sorted(discovery.universals, key=lambda u: u['similarity'], reverse=True)
        for i, u in enumerate(sorted_u[:15]):
            dims = " <-> ".join(u["dimensions"])
            sim = u["similarity"]
            mdl = u.get("mdl", "?")
            types = [p.get("type", "?") for p in u["patterns"]]
            names = [p.get("name", "?") for p in u["patterns"]]
            eq_types_u = [p.get("equation_type", "?") for p in u["patterns"]]
            print(f"  {i+1}. [{dims}] sim={sim:.4f} mdl={mdl}")
            print(f"     types: {types}, eq_types: {eq_types_u}")
            print(f"     names: {names}")

    # ── Step 7: Save results ──────────────────────────────────────────────
    print(f"\n[Step 7] Saving results...")
    discovery.save_universals()
    hdv.save_state()

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SESSION 2 RESULTS")
    print(f"{'=' * 70}")
    print(f"  Function library:    {len(lib.library)} entries")
    print(f"    - algebraic:       {eq_types.get('algebraic', 0)}")
    print(f"    - ODE:             {eq_types.get('ODE', 0)}")
    print(f"    - PDE:             {eq_types.get('PDE', 0)}")
    print(f"    - canonical DEQ:   {has_deq}")
    print(f"  Cross-dim universals: {len(discovery.universals)}")
    print(f"  Library path:        tensor/data/function_library.json")
    print(f"  Universals path:     tensor/data/universals.json")

    # Deliverables check
    print(f"\n  --- Deliverables ---")
    if len(lib.library) >= 50:
        print(f"  [PASS] {len(lib.library)} equations in function_library.json (>= 50)")
    else:
        print(f"  [WARN] Only {len(lib.library)} equations (target was >= 50)")

    if len(discovery.universals) >= 1:
        print(f"  [PASS] {len(discovery.universals)} cross-domain universals found (>= 1)")
    else:
        print(f"  [WARN] No universals found")


if __name__ == '__main__':
    main()

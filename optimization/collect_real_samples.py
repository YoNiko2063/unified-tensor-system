#!/usr/bin/env python3
"""collect_real_samples.py — collect real Rust compilation data and retrain."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

_ROOT = str(Path(__file__).parent.parent)
EXTRACTOR = str(
    Path.home() / "projects/rust-borrow-extractor/target/release/borrow_extractor"
)
RUSTC = str(Path.home() / ".cargo/bin/rustc")
SEARCH_ROOT = str(Path.home() / "projects")
OUTPUT_JSONL = str(Path(_ROOT) / "data/real_samples.jsonl")

EXCLUDE = ["rust-borrow-extractor/src", "/target/", "/.git/"]


def should_skip(path: str) -> bool:
    return any(ex in path for ex in EXCLUDE)


def extract_bv(code: str) -> dict | None:
    """Run extractor on code via stdin, return dict or None on error/parse_error."""
    try:
        result = subprocess.run(
            [EXTRACTOR],
            input=code,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    stdout = result.stdout.strip()
    if not stdout:
        return None

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return None

    if "parse_error" in data:
        return None  # Caller should skip

    return data


def try_compile(path: str) -> tuple[bool, str]:
    """Run rustc --crate-type lib on file, return (success, stderr[:200])."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "out.rlib")
        try:
            result = subprocess.run(
                [
                    RUSTC,
                    "--edition",
                    "2021",
                    "--crate-type",
                    "lib",
                    path,
                    "-o",
                    out_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as exc:
            return False, str(exc)[:200]

    success = result.returncode == 0
    snippet = "" if success else (result.stderr[:200] if result.stderr else "")
    return success, snippet


def collect() -> list[dict]:
    """Scan .rs files, extract BV, compile, log to JSONL."""
    rs_files = []
    for root, dirs, files in os.walk(SEARCH_ROOT):
        # Prune excluded dirs in-place to avoid descending into them
        dirs[:] = [
            d for d in dirs if not should_skip(os.path.join(root, d) + "/")
        ]
        for fname in files:
            if fname.endswith(".rs"):
                full = os.path.join(root, fname)
                if not should_skip(full):
                    rs_files.append(full)

    print(f"Found {len(rs_files)} .rs files to process")

    samples = []
    skipped = 0
    logged = 0
    compile_true = 0
    compile_false = 0

    for path in rs_files:
        print(f"  Processing: {path}")
        try:
            code = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"    [SKIP] read error: {exc}")
            skipped += 1
            continue

        # Run extractor
        bv = extract_bv(code)
        if bv is None:
            print(f"    [SKIP] extractor parse_error or timeout")
            skipped += 1
            continue

        # Run compiler
        compiles, err_snippet = try_compile(path)

        record = {
            "file": path,
            "b1": float(bv.get("b1", 0.0)),
            "b2": float(bv.get("b2", 0.0)),
            "b3": float(bv.get("b3", 0.0)),
            "b4": float(bv.get("b4", 0.0)),
            "b5": float(bv.get("b5", 0.0)),
            "b6": float(bv.get("b6", 0.0)),
            "e_borrow": float(bv.get("e_borrow", 0.0)),
            "compiles": compiles,
            "error_snippet": err_snippet,
        }

        samples.append(record)
        logged += 1
        if compiles:
            compile_true += 1
        else:
            compile_false += 1

        status = "OK" if compiles else "FAIL"
        print(f"    [{status}] e_borrow={record['e_borrow']:.4f}")

    # Write JSONL
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for rec in samples:
            f.write(json.dumps(rec) + "\n")

    print()
    print("=== Real Sample Collection ===")
    print(f"Files scanned: {len(rs_files)}")
    print(f"Files logged: {logged} ({compile_true} compiles, {compile_false} fails)")
    print(f"Files skipped (parse error or timeout): {skipped}")
    print()

    return samples


def retrain(samples: list[dict]):
    """Retrain classifier on real samples, report LOO AUC."""
    if len(samples) < 2:
        print("=== Retraining Report ===")
        print("Insufficient samples for retraining (need >= 2)")
        return

    # Determine if b6 is meaningful (nonzero in any sample)
    has_b6 = any(s.get("b6", 0.0) != 0.0 for s in samples)

    if has_b6:
        feature_names = ["b1", "b2", "b3", "b4", "b5", "b6", "e_borrow"]
        feat_label = "[B1..B6, E_borrow] (7D)"
    else:
        feature_names = ["b1", "b2", "b3", "b4", "b5", "e_borrow"]
        feat_label = "[B1..B5, E_borrow] (6D)"

    X = np.array([[s[f] for f in feature_names] for s in samples], dtype=float)
    y = np.array([1 if s["compiles"] else 0 for s in samples], dtype=int)

    compile_true = int(y.sum())
    compile_false = int((1 - y).sum())

    print("=== Retraining Report ===")
    print(f"Feature vector: {feat_label}")

    if len(set(y.tolist())) < 2:
        print("LOO-CV AUC on real data: N/A - single class")
        print("Synthetic training AUC (reference): 0.918")
        print()
        print("=== Class balance ===")
        print(f"Compile=True:  {compile_true} samples")
        print(f"Compile=False: {compile_false} samples")
        print()
        print("=== D_sep estimate ===")
        print("insufficient data (single class)")
        return

    # LOO cross-validation
    loo = LeaveOneOut()
    y_true_list, y_scores_list = [], []

    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y[train_idx])
        y_scores_list.append(clf.predict_proba(X_test)[0, 1])
        y_true_list.append(y[test_idx][0])

    y_true_arr = np.array(y_true_list)
    y_scores_arr = np.array(y_scores_list)

    if len(set(y_true_list)) > 1:
        auc = roc_auc_score(y_true_arr, y_scores_arr)
        auc_str = f"{auc:.3f}"
    else:
        auc = None
        auc_str = "N/A - single class"

    print(f"LOO-CV AUC on real data: {auc_str}")
    print("Synthetic training AUC (reference): 0.918")
    print()
    print("=== Class balance ===")
    print(f"Compile=True:  {compile_true} samples")
    print(f"Compile=False: {compile_false} samples")
    print()

    # D_sep estimate: threshold scan maximising F1
    print("=== D_sep estimate ===")
    if compile_true < 1 or compile_false < 1:
        print("insufficient data (single class)")
        return

    best_f1 = -1.0
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 99):
        y_pred = (y_scores_arr >= thresh).astype(int)
        if len(set(y_pred.tolist())) < 2:
            continue
        f1 = f1_score(y_true_arr, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    if best_f1 < 0:
        print("insufficient data")
    else:
        print(f"Best F1={best_f1:.3f} at threshold={best_thresh:.2f}")

        # E_borrow statistics per class
        e_true = [s["e_borrow"] for s in samples if s["compiles"]]
        e_false = [s["e_borrow"] for s in samples if not s["compiles"]]
        if e_true and e_false:
            mu_true = float(np.mean(e_true))
            mu_false = float(np.mean(e_false))
            pooled_std = float(
                np.sqrt(
                    (np.var(e_true) * len(e_true) + np.var(e_false) * len(e_false))
                    / (len(e_true) + len(e_false))
                )
            )
            if pooled_std > 0:
                d_sep = abs(mu_true - mu_false) / pooled_std
                print(
                    f"E_borrow: compiles μ={mu_true:.3f}, fails μ={mu_false:.3f}"
                )
                print(f"D_sep (Cohen's d on E_borrow) = {d_sep:.3f}")
            else:
                print("D_sep: pooled_std=0 (cannot compute)")

    print()
    print("=== Per-file summary ===")
    for s in samples:
        status = "COMPILE" if s["compiles"] else "FAIL   "
        print(
            f"  {status}  e_borrow={s['e_borrow']:.4f}  "
            f"b1={s['b1']:.3f} b2={s['b2']:.3f} b3={s['b3']:.3f}  "
            f"{Path(s['file']).name}"
        )


if __name__ == "__main__":
    samples = collect()
    retrain(samples)

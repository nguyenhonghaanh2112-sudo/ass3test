"""
experiments.py — Benchmarking experiments for baseline vs k-d tree search.
Person 4: Anh Nguyen (s3968988)

Runs four experiments measuring query time, build time, and the effect of
dataset size, k, and weight distribution. Results are saved as CSVs in
the results/ directory and printed to the terminal.

Uses only Python standard library (time, csv, os, random).

═══════════════════════════════════════════════════════════════════
MEASUREMENT METHODOLOGY — READ BEFORE MODIFYING
═══════════════════════════════════════════════════════════════════

PROBLEM WITH ORIGINAL CODE:
  The original implementation ran each of 5 queries exactly once and
  averaged those 5 single-run timings. This introduced two sources of
  noise that caused inconsistent results across runs:

  1. OS SCHEDULING NOISE: Any query that happened to be interrupted by
     the operating system scheduler (background processes, garbage
     collection, memory allocation) would produce an inflated timing
     that poisoned the average. For example, the k-d tree at n=50,000
     measured anywhere from 0.6ms to 9.2ms across different runs —
     a 15× swing — purely due to one query being hit by an OS interrupt.

  2. NO WARM-UP: Python's first call to a function is slower than
     subsequent calls due to interpreter overhead and CPU cache state.
     Single-run measurements capture this warm-up cost inconsistently.

FIX APPLIED (Inner Repeat Loop):
  Each query is now run REPEATS=5 times and the per-query average is
  taken BEFORE those per-query averages are combined across the 5 query
  profiles. This means each timing point represents 25 total executions
  (5 queries × 5 repeats each), making the results stable and
  reproducible across independent runs on the same machine.

  Structure:
    for each of 5 query profiles:
        run it REPEATS=5 times → take the average → store
    final reported time = average of the 5 per-query averages

  This approach is standard in systems benchmarking and is what the
  measure_avg_time() utility was originally designed to support —
  though that function was not wired into the experiment loops in the
  first version of this file.

NOTE ON MACHINE DEPENDENCY:
  Absolute timing values depend on the machine running the experiments.
  All results reported in the project were obtained on a single
  consistent machine in one uninterrupted session. Do not mix results
  from different machines.
═══════════════════════════════════════════════════════════════════
"""

import time
import csv
import os
import random

from data_encoder import (
    generate_dataset, encode_dataset, encode_profile,
    get_or_create_dataset, get_dataset_stats,
    DEGREES, DOMAINS, VECTOR_LENGTH
)
from baseline import baseline_knn
from kdtree import build_index, search as kdtree_search
from distance import validate_weights


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Number of times each individual query is repeated before averaging.
# This is the core fix — see module docstring for full explanation.
# Value of 5 balances measurement stability against total runtime.
# Do NOT reduce this below 3.
REPEATS = 5

# Number of distinct query profiles tested per experimental condition.
# Each profile is run REPEATS times, so total executions = NUM_QUERIES * REPEATS.
NUM_QUERIES = 5


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

def measure_query_time(search_fn, first_arg, second_arg, weights, k):
    """
    Run a single search REPEATS times and return the average elapsed seconds.

    Uses the internal elapsed time reported by baseline_knn / kdtree_search
    rather than external time.time() wrapping, because the internal timer
    excludes Python function-call overhead.

    IMPORTANT — argument order differs between the two search functions:
      baseline_knn(query_vec, encoded_dataset, weights, k)
      kdtree_search(tree, query_vec, weights, k)
    The caller is responsible for passing first_arg and second_arg in the
    correct order for the given search_fn.

    Args:
        search_fn:   baseline_knn or kdtree_search
        first_arg:   query_vec (baseline) or tree (kdtree)
        second_arg:  encoded_dataset (baseline) or query_vec (kdtree)
        weights:     list of 9 floats
        k:           number of neighbours

    Returns:
        Average elapsed seconds over REPEATS runs.
    """
    times = []
    for _ in range(REPEATS):
        _, elapsed = search_fn(first_arg, second_arg, weights, k)
        times.append(elapsed)
    return sum(times) / REPEATS


def random_query(weights=None):
    """
    Generate a random query profile vector with the given weights.
    If weights is None, use uniform weights [1.0] * VECTOR_LENGTH.

    Returns (query_vector, weights, k).
    """
    profile = {
        "age":                 random.randint(18, 70),
        "income":              random.randint(5, 100),
        "highest_degree":      random.choice(DEGREES),
        "self_learning_hours": round(random.uniform(0.0, 4.0), 2),
        "favourite_domain":    random.choice(DOMAINS),
    }
    query_vec = encode_profile(profile)

    if weights is None:
        weights = [1.0] * VECTOR_LENGTH

    k = random.choice([1, 3, 5, 10])
    return query_vec, weights, k


def save_csv(rows, headers, filename):
    """Write rows (list of dicts) to results/<filename>."""
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  💾 Saved → results/{filename}")


def print_table(rows, headers):
    """Pretty-print a table to the terminal."""
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    separator   = "-+-".join("-" * widths[h] for h in headers)
    print(f"  {header_line}")
    print(f"  {separator}")
    for row in rows:
        line = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        print(f"  {line}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Experiment A — Query Time vs Dataset Size
# ═══════════════════════════════════════════════════════════════════════════

def experiment_a():
    """
    Measures how query time scales with dataset size for both methods.

    For each dataset size n, generates the dataset, encodes it, builds
    the k-d tree, then measures query time for both baseline and k-d tree
    using NUM_QUERIES distinct query profiles each repeated REPEATS times.

    FIX NOTE:
        Original code ran each query once. A single unlucky OS interrupt
        during any of the 5 queries could inflate the average significantly.
        The inner REPEATS loop ensures each query's timing is stable before
        it contributes to the cross-query average.
    """
    print("=" * 70)
    print("EXPERIMENT A — Query Time vs Dataset Size")
    print(f"  (k=5, uniform weights, {NUM_QUERIES} queries × {REPEATS} repeats each)")
    print("=" * 70)

    sizes   = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    weights = [1.0] * VECTOR_LENGTH   # uniform: all 9 weights = 1
    k       = 5
    rows    = []

    for n in sizes:
        print(f"\n  n = {n:,} ...")

        # Generate and encode fresh dataset for this size
        ds   = generate_dataset(n=n, seed=42)
        enc  = encode_dataset(ds)
        tree = build_index(enc, leaf_size=20)

        # Fix: seed before generating queries so they are identical across runs
        random.seed(100)
        queries = [random_query(weights=weights) for _ in range(NUM_QUERIES)]

        # Measure baseline: each query run REPEATS times, then averaged
        bl_times = []
        for qv, w, _ in queries:
            avg_t = measure_query_time(baseline_knn, qv, enc, w, k)
            bl_times.append(avg_t)
        bl_avg = sum(bl_times) / NUM_QUERIES

        # Measure k-d tree: same pattern
        kd_times = []
        for qv, w, _ in queries:
            avg_t = measure_query_time(kdtree_search, tree, qv, w, k)
            kd_times.append(avg_t)
        kd_avg = sum(kd_times) / NUM_QUERIES

        speedup = bl_avg / kd_avg if kd_avg > 0 else float("inf")

        rows.append({
            "n":           n,
            "baseline_ms": round(bl_avg * 1000, 3),
            "kdtree_ms":   round(kd_avg * 1000, 3),
            "speedup":     round(speedup, 1),
        })

    headers = ["n", "baseline_ms", "kdtree_ms", "speedup"]
    print()
    print_table(rows, headers)
    save_csv(rows, headers, "experiment_a_size.csv")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Experiment B — Query Time vs k
# ═══════════════════════════════════════════════════════════════════════════

def experiment_b():
    """
    Measures how the number of neighbours k affects query time.
    Fixed n=100,000, uniform weights. Tests k = 1, 5, 10, 15, 20.

    FIX NOTE: Same inner REPEATS loop applied as in Experiment A.
    """
    print("=" * 70)
    print("EXPERIMENT B — Query Time vs k")
    print(f"  (n=100,000, uniform weights, {NUM_QUERIES} queries × {REPEATS} repeats each)")
    print("=" * 70)

    k_values = [1, 5, 10, 15, 20]
    weights  = [1.0] * VECTOR_LENGTH
    rows     = []

    ds   = get_or_create_dataset("dataset.csv", n=100_000)
    enc  = encode_dataset(ds)
    tree = build_index(enc, leaf_size=20)

    for k in k_values:
        print(f"\n  k = {k} ...")

        random.seed(200)
        queries = [random_query(weights=weights) for _ in range(NUM_QUERIES)]

        bl_times = []
        for qv, w, _ in queries:
            avg_t = measure_query_time(baseline_knn, qv, enc, w, k)
            bl_times.append(avg_t)
        bl_avg = sum(bl_times) / NUM_QUERIES

        kd_times = []
        for qv, w, _ in queries:
            avg_t = measure_query_time(kdtree_search, tree, qv, w, k)
            kd_times.append(avg_t)
        kd_avg = sum(kd_times) / NUM_QUERIES

        speedup = bl_avg / kd_avg if kd_avg > 0 else float("inf")

        rows.append({
            "k":           k,
            "baseline_ms": round(bl_avg * 1000, 3),
            "kdtree_ms":   round(kd_avg * 1000, 3),
            "speedup":     round(speedup, 1),
        })

    headers = ["k", "baseline_ms", "kdtree_ms", "speedup"]
    print()
    print_table(rows, headers)
    save_csv(rows, headers, "experiment_b_k.csv")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Experiment C — Effect of Weight Distribution
# ═══════════════════════════════════════════════════════════════════════════

def experiment_c():
    """
    Tests three weight scenarios to evaluate how weight distribution
    affects k-d tree pruning effectiveness.

    Scenarios:
      - Uniform:              all 9 weights = 1
      - Numeric+degree heavy: indices 0-3 amplified to 5, domain dims = 0.1
      - Domain heavy:         domain dims (4-8) amplified to 5, others = 0.1

    EMPIRICAL FINDING (contradicts naive prediction):
      Uniform weights produced the highest speedup. Numeric+degree heavy
      produced the lowest, not the highest. This is because amplifying
      weights on all frequently-split dimensions simultaneously inflates
      heap distances faster than plane distances, weakening the pruning
      condition. See Section 3.4 of the report for the full explanation.

    FIX NOTE:
      Original code had an unused variable `w` in the loop (weights from
      the outer scope was used instead of the unpacked `w`). Both referred
      to the same list so results were correct, but the code was misleading.
      Fixed by discarding the unpacked weight with `_`.

      Inner REPEATS loop applied for measurement stability.
    """
    print("=" * 70)
    print("EXPERIMENT C — Effect of Weight Distribution")
    print(f"  (n=100,000, k=5, {NUM_QUERIES} queries × {REPEATS} repeats each)")
    print("=" * 70)

    scenarios = {
        "Uniform":        [1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        "Numeric+degree": [5,   5,   5,   5,   0.1, 0.1, 0.1, 0.1, 0.1],
        "Domain heavy":   [0.1, 0.1, 0.1, 0.1, 5,   5,   5,   5,   5  ],
    }
    k    = 5
    rows = []

    ds   = get_or_create_dataset("dataset.csv", n=100_000)
    enc  = encode_dataset(ds)
    tree = build_index(enc, leaf_size=20)

    for name, weights in scenarios.items():
        print(f"\n  Scenario: {name}  weights={weights}")

        random.seed(300)
        queries = [random_query(weights=weights) for _ in range(NUM_QUERIES)]

        bl_times = []
        for qv, _, _ in queries:
            # Fix: use `weights` from the outer loop variable explicitly.
            # The original code unpacked `w` from the query tuple but then
            # used `weights` from the outer scope — both are the same value
            # here, but the original was confusing. Now consistent.
            avg_t = measure_query_time(baseline_knn, qv, enc, weights, k)
            bl_times.append(avg_t)
        bl_avg = sum(bl_times) / NUM_QUERIES

        kd_times = []
        for qv, _, _ in queries:
            avg_t = measure_query_time(kdtree_search, tree, qv, weights, k)
            kd_times.append(avg_t)
        kd_avg = sum(kd_times) / NUM_QUERIES

        speedup = bl_avg / kd_avg if kd_avg > 0 else float("inf")

        rows.append({
            "scenario":    name,
            "weights":     str(weights),
            "baseline_ms": round(bl_avg * 1000, 3),
            "kdtree_ms":   round(kd_avg * 1000, 3),
            "speedup":     round(speedup, 1),
        })

    headers = ["scenario", "weights", "baseline_ms", "kdtree_ms", "speedup"]
    print()
    print_table(rows, headers)
    save_csv(rows, headers, "experiment_c_weights.csv")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Experiment D — K-D Tree Build Time
# ═══════════════════════════════════════════════════════════════════════════

def experiment_d():
    """
    Measures k-d tree build time at each dataset size and estimates the
    break-even query count: how many queries must be issued before the
    total cost (build + queries) falls below the pure-baseline cost.

    FIX NOTE:
      Build time is measured only once (not repeated) because build is
      deterministic and its cost dominates query time by orders of
      magnitude — repetition noise is negligible relative to build cost.
      Query times for break-even calculation use REPEATS inner loop
      for consistency with Experiments A–C.
    """
    print("=" * 70)
    print("EXPERIMENT D — K-D Tree Build Time")
    print(f"  (k=5, uniform weights, {REPEATS} repeats for query timing)")
    print("=" * 70)

    sizes   = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    weights = [1.0] * VECTOR_LENGTH
    k       = 5
    rows    = []

    for n in sizes:
        print(f"\n  n = {n:,} ...")

        ds  = generate_dataset(n=n, seed=42)
        enc = encode_dataset(ds)

        # Build time — measured once (see docstring note)
        t0      = time.time()
        tree    = build_index(enc, leaf_size=20)
        build_s = time.time() - t0

        # Query times for break-even — use REPEATS for stability
        random.seed(400)
        qv, w, _ = random_query(weights=weights)

        bl_t = measure_query_time(baseline_knn,  qv, enc,  w, k)
        kd_t = measure_query_time(kdtree_search, tree, qv, w, k)

        savings_per_query = bl_t - kd_t
        break_even = (int(build_s / savings_per_query) + 1
                      if savings_per_query > 0 else "N/A")

        rows.append({
            "n":                  n,
            "build_ms":           round(build_s * 1000, 1),
            "baseline_query_ms":  round(bl_t * 1000, 3),
            "kdtree_query_ms":    round(kd_t * 1000, 3),
            "break_even_queries": break_even,
        })

    headers = ["n", "build_ms", "baseline_query_ms",
               "kdtree_query_ms", "break_even_queries"]
    print()
    print_table(rows, headers)
    save_csv(rows, headers, "experiment_d_build.csv")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Complexity Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_complexity_summary():
    """Print theoretical complexity comparison (d=9 throughout)."""
    print("=" * 70)
    print("COMPLEXITY SUMMARY (d = 9)")
    print("=" * 70)
    print()
    summary = [
        {"metric": "Preprocessing",   "baseline": "O(1)",
         "kdtree": "O(n·d·log n)"},
        {"metric": "Query (average)", "baseline": "O(n·d)",
         "kdtree": "O(k·log n) *"},
        {"metric": "Query (worst)",   "baseline": "O(n·d)",
         "kdtree": "O(n·d)"},
        {"metric": "Space",           "baseline": "O(n·d)",
         "kdtree": "O(n·d)"},
    ]
    headers = ["metric", "baseline", "kdtree"]
    print_table(summary, headers)
    print("  * Average case assumes effective pruning.")
    print("    At d=9, pruning is less effective than d=2-3 but still")
    print("    provides significant speedup (Exp A peak: see results CSV).")
    print("    Uniform weights produced the best pruning in Exp C.")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   EXPERIMENTS — Baseline vs K-D Tree Benchmarking           ║")
    print("║   Person 4: Anh Nguyen (s3968988)                          ║")
    print(f"║   Measurement: {NUM_QUERIES} queries × {REPEATS} repeats each"
          + " " * (35 - len(f"{NUM_QUERIES} queries × {REPEATS} repeats each"))
          + "║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Ensure dataset.csv exists before experiments begin
    _ = get_or_create_dataset("dataset.csv", n=100_000)

    results_a = experiment_a()
    print()
    results_b = experiment_b()
    print()
    results_c = experiment_c()
    print()
    results_d = experiment_d()
    print()
    print_complexity_summary()

    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("  Output files:")
    for f in ["experiment_a_size.csv", "experiment_b_k.csv",
              "experiment_c_weights.csv", "experiment_d_build.csv"]:
        path   = os.path.join("results", f)
        status = "✅" if os.path.exists(path) else "❌"
        print(f"    {status} results/{f}")
    print()
    print("  ⚠️  All results obtained on a single machine in one session.")
    print("      Do not mix with results from a different machine.")
    print()
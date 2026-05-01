# =============================================================================
# FILE: main.py
# OWNER: PERSON 2 — System Integration & Entry Point
# PURPOSE: Tie all modules together into a working similarity search system.
#          Supports three modes: dataset generation, interactive queries,
#          and batch queries from a JSON file.
#
# SYSTEM ARCHITECTURE:
#
#   data_encoder.py → generates + encodes dataset (Person 1)
#          ↓
#   kdtree.py       → builds tree index, one-time cost (Person 3)
#          ↓
#   User query → encode_profile() → [baseline.py] → top-k results (brute-force)
#                                  → [kdtree.py]  → top-k results (tree search)
#                                  → compare + display both
#
# USAGE:
#   python main.py --generate           # Generate dataset only
#   python main.py                      # Interactive query mode
#   python main.py --batch queries.json # Batch mode for experiments
#
# The key design decision is to run BOTH search methods on every query
# and compare their results. This lets us verify KD-tree correctness
# (same neighbours as brute-force) while also measuring the speed
# difference that justifies the tree's added complexity.
#
# STDLIB ONLY — no external packages.
# =============================================================================

import sys
import json

from data_encoder import (
    DEGREES, DOMAINS, VECTOR_LENGTH,
    encode_profile, encode_dataset,
    get_or_create_dataset, validate_profile,
    decode_vector, get_dataset_stats,
)
from distance import weighted_distance, validate_weights
from baseline import (
    baseline_knn,
    parse_query_profile, parse_k, parse_weights,
    display_results, compare_results,
)

# ─────────────────────────────────────────────────────────────────────────────
# We import from Person 3's module. If kdtree.py is not yet delivered,
# we provide a graceful fallback so baseline-only testing can proceed.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from kdtree import build_index, search as kdtree_search
    KDTREE_AVAILABLE = True
except ImportError:
    KDTREE_AVAILABLE = False
    print("⚠️  kdtree.py not found — running baseline-only mode.")
    print("    Person 3 needs to deliver kdtree.py for full functionality.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — Load data, encode vectors, build tree
# ─────────────────────────────────────────────────────────────────────────────

def startup(dataset_path="dataset.csv", n=100_000):
    """
    Initialise the system by loading data and building search structures.

    Steps:
        1. Load (or generate) the dataset from CSV via Person 1's module.
        2. Encode all profiles into 9-dim vectors for distance computation.
        3. Build the KD-tree index for fast nearest-neighbour queries
           (if Person 3's module is available).

    This runs once at program start. The encoded dataset and tree are then
    reused for every query, avoiding repeated work.

    Args:
        dataset_path: Path to the CSV file (default "dataset.csv").
        n:            Number of profiles to generate if CSV doesn't exist.

    Returns:
        Tuple of (dataset, encoded_dataset, tree).
        tree is None if kdtree.py is not available.
    """
    print("=" * 60)
    print("  Similarity Search System — Startup")
    print("=" * 60)

    # Step 1: Load raw profiles from CSV (or generate + save if first run)
    dataset = get_or_create_dataset(dataset_path, n=n)
    print(f"  Dataset: {len(dataset):,} profiles loaded")

    # Step 2: Pre-encode all profiles into numerical vectors.
    # This is done once so we don't re-encode on every query.
    encoded_dataset = encode_dataset(dataset)

    # Step 3: Build the KD-tree (if available)
    tree = None
    if KDTREE_AVAILABLE:
        tree = build_index(encoded_dataset)
        print("  KD-tree: built successfully")
    else:
        print("  KD-tree: UNAVAILABLE (kdtree.py not found)")

    print("=" * 60)
    return dataset, encoded_dataset, tree


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE — One query at a time via CLI prompts
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode(dataset, encoded_dataset, tree):
    """
    Interactive query loop: prompt user, run both searches, compare results.

    Each iteration:
        1. Collect query profile, k, and weights from user input
        2. Encode the query into a 9-dim vector
        3. Run baseline (brute-force) search
        4. Run KD-tree search (if available)
        5. Display results from both methods
        6. Compare to verify KD-tree correctness
        7. Print timing summary

    The loop continues until the user types 'quit' at the age prompt.
    """
    print("\n  Type 'quit' at any prompt to exit.\n")

    while True:
        try:
            # Collect query parameters from user
            query_profile = parse_query_profile()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or piped input gracefully
            print("\n  Exiting...")
            break

        # Check for quit signal (parse_query_profile returns None is not used here,
        # but the user can Ctrl+C to exit)

        k = parse_k()
        weights = parse_weights()

        # Encode the user's query profile into a 9-dim vector
        query_vector = encode_profile(query_profile)

        print(f"\n  Query vector: {[round(v, 4) for v in query_vector]}")
        print(f"  Searching for k={k} nearest neighbours...")

        # ── Run baseline (brute-force) search ────────────────────────────
        bl_results, bl_time = baseline_knn(
            query_vector, encoded_dataset, weights, k)
        display_results(bl_results, "Baseline (Linear Scan)", bl_time, dataset)

        # ── Run KD-tree search (if available) ────────────────────────────
        if KDTREE_AVAILABLE and tree is not None:
            kd_results, kd_time = kdtree_search(
                tree, query_vector, weights, k)
            display_results(kd_results, "KD-Tree", kd_time, dataset)

            # ── Compare results from both methods ────────────────────────
            compare_results(bl_results, kd_results)

            # ── Timing summary ───────────────────────────────────────────
            speedup = bl_time / kd_time if kd_time > 0 else float("inf")
            print(f"\n  ⏱️  Baseline: {bl_time:.4f}s | KD-Tree: {kd_time:.4f}s "
                  f"| Speedup: {speedup:.1f}×")
        else:
            print("\n  (KD-tree not available — showing baseline results only)")

        # Ask if user wants another query
        again = input("\n  Another query? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            break

    print("\n  Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH MODE — Run queries from a JSON file (for Person 4's experiments)
# ─────────────────────────────────────────────────────────────────────────────

def batch_mode(dataset, encoded_dataset, tree, batch_file):
    """
    Load queries from a JSON file and run both search methods on each.

    Expected JSON format — a list of query objects:
        [
          {
            "profile": {"age": 25, "income": 30, ...},
            "k": 5,
            "weights": [1, 1, 1, 1, 1, 1, 1, 1, 1]
          },
          ...
        ]

    For each query, we:
        1. Validate the profile and weights
        2. Run baseline search
        3. Run KD-tree search (if available)
        4. Compare results and record timing

    At the end, print a summary table with timing and correctness for
    each query — useful for the Section 4 performance analysis.

    Args:
        dataset:         Raw dataset list.
        encoded_dataset: List of (id, vector) tuples.
        tree:            KD-tree root node (or None).
        batch_file:      Path to the JSON file containing queries.
    """
    # Load queries from JSON
    try:
        with open(batch_file, "r") as f:
            queries = json.load(f)
    except FileNotFoundError:
        print(f"  ❌ File not found: {batch_file}")
        return
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON in {batch_file}: {e}")
        return

    print(f"\n  Loaded {len(queries)} queries from {batch_file}")
    print("=" * 70)

    # Track results for summary table
    summary = []

    for i, q in enumerate(queries):
        print(f"\n  ── Query {i + 1}/{len(queries)} ──")

        # Extract and validate query components
        try:
            profile = q["profile"]
            k = q["k"]
            weights = [float(w) for w in q["weights"]]

            validate_profile(profile)
            validate_weights(weights, expected_length=VECTOR_LENGTH)

            if not (1 <= k <= 20):
                raise ValueError(f"k must be 1–20, got {k}")

        except (KeyError, ValueError) as e:
            print(f"  ❌ Skipping query {i + 1}: {e}")
            summary.append({
                "query": i + 1, "status": "SKIPPED", "error": str(e)})
            continue

        # Encode the query profile
        query_vector = encode_profile(profile)

        # Run baseline search
        bl_results, bl_time = baseline_knn(
            query_vector, encoded_dataset, weights, k)
        display_results(bl_results, "Baseline", bl_time, dataset)

        # Run KD-tree search (if available)
        kd_time = None
        match = None
        if KDTREE_AVAILABLE and tree is not None:
            kd_results, kd_time = kdtree_search(
                tree, query_vector, weights, k)
            display_results(kd_results, "KD-Tree", kd_time, dataset)

            # Check if both methods agree
            bl_ids = [r["profile_id"] for r in bl_results]
            kd_ids = [r["profile_id"] for r in kd_results]
            match = (bl_ids == kd_ids)
            compare_results(bl_results, kd_results)

        summary.append({
            "query": i + 1,
            "status": "OK",
            "k": k,
            "bl_time": bl_time,
            "kd_time": kd_time,
            "match": match,
        })

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  BATCH SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Query':<8} {'k':<5} {'Baseline':<12} {'KD-Tree':<12} "
          f"{'Speedup':<10} {'Match'}")
    print(f"  {'─' * 60}")

    for s in summary:
        if s["status"] == "SKIPPED":
            print(f"  {s['query']:<8} SKIPPED — {s['error']}")
            continue

        bl_str = f"{s['bl_time']:.4f}s"
        kd_str = f"{s['kd_time']:.4f}s" if s['kd_time'] is not None else "N/A"

        if s['kd_time'] is not None and s['kd_time'] > 0:
            speedup = f"{s['bl_time'] / s['kd_time']:.1f}×"
        else:
            speedup = "N/A"

        match_str = "✅" if s['match'] is True else (
            "⚠️" if s['match'] is False else "—")

        print(f"  {s['query']:<8} {s['k']:<5} {bl_str:<12} {kd_str:<12} "
              f"{speedup:<10} {match_str}")

    print(f"{'═' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse command-line arguments for mode selection
    if "--generate" in sys.argv:
        # Generate-only mode: create dataset.csv and print stats, then exit
        print("  Mode: Generate dataset only\n")
        dataset = get_or_create_dataset("dataset.csv", n=100_000)
        stats = get_dataset_stats(dataset)
        print(f"\n  Dataset: {stats['n']:,} profiles")
        print(f"  Age:    {stats['age_min']}–{stats['age_max']}  "
              f"(mean {stats['age_mean']})")
        print(f"  Income: {stats['income_min']}–{stats['income_max']}  "
              f"(mean {stats['income_mean']})")
        print("  Done. dataset.csv is ready.")
        sys.exit(0)

    # Normal startup — load data, encode, build tree
    dataset, encoded_dataset, tree = startup()

    if "--batch" in sys.argv:
        # Batch mode: run queries from a JSON file
        try:
            idx = sys.argv.index("--batch")
            batch_file = sys.argv[idx + 1]
        except (IndexError, ValueError):
            print("  ❌ Usage: python main.py --batch <queries.json>")
            sys.exit(1)

        print(f"  Mode: Batch ({batch_file})\n")
        batch_mode(dataset, encoded_dataset, tree, batch_file)
    else:
        # Interactive mode: prompt user for queries
        print("  Mode: Interactive\n")
        interactive_mode(dataset, encoded_dataset, tree)
# =============================================================================
# FILE: baseline.py
# OWNER: PERSON 2 — Baseline Linear Scan Search
# PURPOSE: Provide a brute-force k-nearest-neighbour search that scans every
#          profile in the dataset. This serves as the correctness benchmark
#          against which Person 3's KD-tree results are compared.
#
# WHY BRUTE-FORCE?
#     A linear scan is the simplest possible approach to k-NN: compute the
#     distance from the query to every single point, sort, and return the
#     top k. It is guaranteed to find the exact nearest neighbours because
#     it checks everything — no pruning, no approximation.
#
#     The downside is performance: O(n × d) distance computations plus
#     O(n log n) sorting. For 100,000 profiles with d=9, this is fine for
#     a single query but becomes slow for bulk experiments. That is exactly
#     why Person 3 builds a KD-tree — to reduce the average case.
#
#     By running BOTH methods on every query and comparing results, we can
#     verify that the KD-tree returns the same neighbours as brute-force,
#     confirming its correctness.
#
# STDLIB ONLY — no external packages.
# =============================================================================

import time

from distance import weighted_distance, validate_weights
from data_encoder import (
    DEGREES, DOMAINS, VECTOR_LENGTH,
    encode_profile, validate_profile, decode_vector,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def baseline_knn(query_vector, encoded_dataset, weights, k):
    """
    Brute-force k-nearest-neighbour search via linear scan.

    Algorithm:
        1. Iterate through every (profile_id, vector) pair in the dataset
        2. Compute weighted Euclidean distance from the query to each vector
        3. Sort all distances in ascending order
        4. Return the top k closest profiles

    Complexity:
        Time:  O(n × d + n log n) where n = dataset size, d = 9 dimensions
               The n×d term is for computing all distances; n log n for sorting.
        Space: O(n) to store the distance list.

    Args:
        query_vector:    List of 9 floats — the encoded query profile.
        encoded_dataset: List of (profile_id, vector) tuples from encode_dataset().
        weights:         List of 9 non-negative floats — per-dimension weights.
        k:               Number of nearest neighbours to return (1–20).

    Returns:
        Tuple of (results, elapsed_seconds) where results is a list of
        {"profile_id": int, "distance": float} dicts sorted by distance.

    Raises:
        ValueError: if weights are invalid or k is out of range.
    """
    # Validate inputs before doing any expensive computation
    validate_weights(weights)
    if not (1 <= k <= 20):
        raise ValueError(f"k must be 1–20, got {k}")

    start = time.time()

    # Compute distance from query to every profile in the dataset.
    # We store (profile_id, distance) pairs for sorting.
    distances = []
    for pid, vec in encoded_dataset:
        d = weighted_distance(query_vector, vec, weights)
        distances.append((pid, d))

    # Sort by distance ascending — closest profiles first.
    # Python's Timsort is O(n log n) in the worst case.
    distances.sort(key=lambda x: x[1])

    elapsed = time.time() - start

    # Take only the top k and round distances for clean display
    results = [{"profile_id": pid, "distance": round(dist, 6)}
               for pid, dist in distances[:k]]

    return results, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# CLI INPUT FUNCTIONS
# Each uses a while-True retry loop so invalid input never crashes the program.
# ─────────────────────────────────────────────────────────────────────────────

def parse_query_profile():
    """
    Interactively prompt the user for all 5 profile attributes.

    Each attribute is validated individually with a retry loop, so a typo
    on one field does not force the user to re-enter everything. After all
    fields are collected, we run validate_profile() as a final safety check
    before the dict ever reaches encode_profile().

    Returns:
        A valid profile dict (without 'id' — not needed for queries).
    """
    print("\n── Enter Query Profile ─────────────────────────────────────")

    # ── Age ───────────────────────────────────────────────────────────────
    while True:
        try:
            age = int(input("  Age (18–70): "))
            if 18 <= age <= 70:
                break
            print("    ⚠️  Must be between 18 and 70.")
        except ValueError:
            print("    ⚠️  Please enter a whole number.")

    # ── Income ────────────────────────────────────────────────────────────
    while True:
        try:
            income = int(input("  Income (5–100, thousands): "))
            if 5 <= income <= 100:
                break
            print("    ⚠️  Must be between 5 and 100.")
        except ValueError:
            print("    ⚠️  Please enter a whole number.")

    # ── Highest Degree ────────────────────────────────────────────────────
    # Show numbered options so the user doesn't have to type the exact string
    while True:
        print("  Highest Degree:")
        for i, deg in enumerate(DEGREES):
            print(f"    {i + 1}. {deg}")
        try:
            choice = int(input("  Enter number (1–4): "))
            if 1 <= choice <= len(DEGREES):
                degree = DEGREES[choice - 1]
                break
            print("    ⚠️  Choose 1–4.")
        except ValueError:
            print("    ⚠️  Please enter a number.")

    # ── Self-Learning Hours ───────────────────────────────────────────────
    while True:
        try:
            hours = float(input("  Self-learning hours/day (0.0–4.0): "))
            if 0.0 <= hours <= 4.0:
                hours = round(hours, 2)
                break
            print("    ⚠️  Must be between 0.0 and 4.0.")
        except ValueError:
            print("    ⚠️  Please enter a number.")

    # ── Favourite Domain ──────────────────────────────────────────────────
    while True:
        print("  Favourite Domain:")
        for i, dom in enumerate(DOMAINS):
            print(f"    {i + 1}. {dom}")
        try:
            choice = int(input("  Enter number (1–5): "))
            if 1 <= choice <= len(DOMAINS):
                domain = DOMAINS[choice - 1]
                break
            print("    ⚠️  Choose 1–5.")
        except ValueError:
            print("    ⚠️  Please enter a number.")

    profile = {
        "age": age,
        "income": income,
        "highest_degree": degree,
        "self_learning_hours": hours,
        "favourite_domain": domain,
    }

    # Final validation — catches any edge cases we might have missed
    validate_profile(profile)
    return profile


def parse_k():
    """
    Prompt the user for k (number of nearest neighbours to find).

    We cap k at 20 to keep the output readable and because returning
    more than 20 similar profiles is rarely useful in practice.

    Returns:
        An integer between 1 and 20.
    """
    while True:
        try:
            k = int(input("\n  How many neighbours k (1–20): "))
            if 1 <= k <= 20:
                return k
            print("    ⚠️  Must be between 1 and 20.")
        except ValueError:
            print("    ⚠️  Please enter a whole number.")


def parse_weights():
    """
    Prompt the user for 9 space-separated weight values.

    We display dimension labels so the user knows which weight controls
    which attribute. This is important because the vector layout is not
    obvious — index 3 is degree (label encoded), indices 4–8 are the
    five domain one-hot dimensions.

    Dimension labels:
        [0] age   [1] income   [2] hours   [3] degree
        [4] AI    [5] SoftEng  [6] DataSci [7] Cyber  [8] BizAnal

    Example inputs:
        Equal weight:     1 1 1 1 1 1 1 1 1
        Age+degree focus: 5 0 0 5 0 0 0 0 0

    Returns:
        A validated list of 9 non-negative floats.
    """
    print(f"\n  Enter {VECTOR_LENGTH} weights (space-separated):")
    print("    [0] age   [1] income   [2] hours   [3] degree")
    print("    [4] AI    [5] SoftEng  [6] DataSci [7] Cyber  [8] BizAnal")
    print("    Example: 1 1 1 1 1 1 1 1 1")

    while True:
        try:
            raw = input("  Weights: ").strip().split()
            weights = [float(w) for w in raw]
            validate_weights(weights, expected_length=VECTOR_LENGTH)
            return weights
        except ValueError as e:
            print(f"    ⚠️  {e}")
            print(f"    Please enter exactly {VECTOR_LENGTH} non-negative numbers.")


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY AND COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def display_results(results, method, elapsed, dataset):
    """
    Print the top-k results in a readable table format.

    For each result, we look up the original profile from the dataset
    to show human-readable attributes (age, degree, domain) alongside
    the profile ID and computed distance. This helps the user verify
    that the results make intuitive sense.

    Args:
        results: List of {"profile_id": int, "distance": float} dicts.
        method:  String label — e.g. "Baseline" or "KD-Tree".
        elapsed: Search time in seconds.
        dataset: The raw dataset list (for looking up profile details).
    """
    # Build a quick lookup dict: profile_id → profile dict.
    # This avoids scanning the full dataset for every result row.
    id_lookup = {p["id"]: p for p in dataset}

    print(f"\n{'═' * 70}")
    print(f"  {method} Results  (took {elapsed:.4f}s)")
    print(f"{'═' * 70}")
    print(f"  {'Rank':<5} {'ID':<8} {'Distance':<12} {'Age':<5} "
          f"{'Income':<8} {'Degree':<15} {'Domain'}")
    print(f"  {'─' * 65}")

    for rank, r in enumerate(results, start=1):
        pid = r["profile_id"]
        dist = r["distance"]
        p = id_lookup.get(pid, {})

        # Display the original profile attributes for interpretability
        age    = p.get("age", "?")
        income = p.get("income", "?")
        degree = p.get("highest_degree", "?")
        domain = p.get("favourite_domain", "?")

        print(f"  {rank:<5} {pid:<8} {dist:<12.6f} {age:<5} "
              f"{income:<8} {degree:<15} {domain}")

    print(f"{'═' * 70}")


def compare_results(baseline_results, kdtree_results):
    """
    Compare the profile IDs returned by baseline and KD-tree searches.

    If both methods return the same set of profile IDs (in the same order),
    that confirms the KD-tree is finding the exact same nearest neighbours
    as the brute-force scan — which is the correctness guarantee we need.

    If they differ, we print the specific differences so the team can debug.
    Minor ordering differences at equal distances are acceptable but flagged.

    Args:
        baseline_results: List of {"profile_id", "distance"} from baseline.
        kdtree_results:   List of {"profile_id", "distance"} from KD-tree.
    """
    bl_ids = [r["profile_id"] for r in baseline_results]
    kd_ids = [r["profile_id"] for r in kdtree_results]

    if bl_ids == kd_ids:
        print("\n  ✅ Both methods returned identical results.")
    else:
        print("\n  ⚠️  Results differ between Baseline and KD-Tree:")
        print(f"    Baseline IDs: {bl_ids}")
        print(f"    KD-Tree  IDs: {kd_ids}")

        # Check if same set but different order (tie-breaking difference)
        if set(bl_ids) == set(kd_ids):
            print("    Note: Same profiles, different order "
                  "(likely a tie-breaking difference).")
        else:
            # Find which IDs are in one but not the other
            only_bl = set(bl_ids) - set(kd_ids)
            only_kd = set(kd_ids) - set(bl_ids)
            if only_bl:
                print(f"    Only in Baseline: {only_bl}")
            if only_kd:
                print(f"    Only in KD-Tree:  {only_kd}")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS — run with: python baseline.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_encoder import generate_dataset, encode_dataset

    print("Running baseline.py self-tests...\n")

    # Generate a small test dataset
    ds = generate_dataset(n=500, seed=0)
    encoded = encode_dataset(ds)

    # Create a query from the first profile in the dataset
    query = encode_profile(ds[0])
    weights = [1.0] * VECTOR_LENGTH

    # Test 1: k=1 should return the query profile itself (distance ≈ 0)
    results, elapsed = baseline_knn(query, encoded, weights, k=1)
    assert len(results) == 1
    assert results[0]["profile_id"] == 0
    assert results[0]["distance"] < 1e-9
    print(f"  ✅ Test 1 PASSED: k=1 returns query itself (d={results[0]['distance']})")

    # Test 2: k=5 returns 5 results, all sorted by distance
    results, elapsed = baseline_knn(query, encoded, weights, k=5)
    assert len(results) == 5
    for i in range(len(results) - 1):
        assert results[i]["distance"] <= results[i + 1]["distance"]
    print(f"  ✅ Test 2 PASSED: k=5 returns 5 sorted results")

    # Test 3: Invalid k should raise ValueError
    try:
        baseline_knn(query, encoded, weights, k=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Test 3 PASSED: k=0 rejected")

    try:
        baseline_knn(query, encoded, weights, k=21)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Test 4 PASSED: k=21 rejected")

    # Test 5: Zero-weight on all domain dims should ignore domain differences
    w_no_domain = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results, _ = baseline_knn(query, encoded, w_no_domain, k=5)
    assert len(results) == 5
    print("  ✅ Test 5 PASSED: Search works with domain weights zeroed out")

    # Test 6: Display and compare functions don't crash
    display_results(results, "Baseline Test", elapsed, ds)
    compare_results(results, results)
    print("  ✅ Test 6 PASSED: display_results and compare_results work")

    print(f"\nAll baseline.py tests PASSED ✅")
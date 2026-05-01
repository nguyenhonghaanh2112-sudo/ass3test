# =============================================================================
# FILE: distance.py
# OWNER: PERSON 2 — SHARED WITH PERSON 3
# PURPOSE: Provide a weighted Euclidean distance function and weight validation
#          used by both the baseline linear scan and the KD-tree search.
#
# ⚠️  Do NOT modify without notifying Person 3 — their KD-tree traversal
#     calls weighted_distance() on every node comparison.
#
# DISTANCE FORMULA:
#     d(A, B) = sqrt( Σ w_i × (A_i − B_i)² )    for i = 0..8
#
# WHY WEIGHTED EUCLIDEAN (not plain Euclidean)?
#     Plain Euclidean treats every dimension equally. But in our dataset,
#     a user might care more about matching age and degree than about
#     matching domain. Weights let the user control how much each
#     attribute influences the similarity ranking.
#
#     For example, weights = [5, 5, 1, 5, 0, 0, 0, 0, 0] would focus
#     the search on age, income, and degree while ignoring domain entirely.
#
# WHY sqrt AND NOT SQUARED DISTANCE?
#     For ranking purposes, squared distance gives identical ordering.
#     But we include sqrt so the displayed distances are in the same
#     unit scale as the input features — easier for users to interpret
#     and necessary for the report's worked examples.
#
# VECTOR DIMENSIONS (9 total):
#     [0] age   [1] income   [2] hours   [3] degree
#     [4] AI    [5] SoftEng  [6] DataSci [7] Cyber  [8] BizAnal
# =============================================================================

import math


def weighted_distance(vec_a, vec_b, weights):
    """
    Compute the weighted Euclidean distance between two 9-dim vectors.

    Formula: sqrt( Σ w_i × (A_i − B_i)² )

    The weight for each dimension scales its contribution to the total
    distance. A weight of 0 effectively ignores that dimension; a weight
    of 5 makes it five times more influential than a weight of 1.

    Args:
        vec_a:   List of 9 floats — the query vector.
        vec_b:   List of 9 floats — a dataset vector.
        weights: List of 9 non-negative floats — per-dimension weights.

    Returns:
        A single non-negative float representing the distance.

    Raises:
        ValueError: if the three lists have mismatched lengths.
    """
    # All three lists must be the same length.
    # We check this explicitly rather than letting a silent index error
    # produce a confusing traceback deep in the loop.
    if len(vec_a) != len(vec_b) or len(vec_a) != len(weights):
        raise ValueError(
            f"Length mismatch: vec_a={len(vec_a)}, vec_b={len(vec_b)}, "
            f"weights={len(weights)} — all must be 9"
        )

    # Accumulate the weighted squared differences.
    # We avoid creating intermediate lists — a running sum is both
    # simpler and faster, especially when called 100,000 times per query.
    total = 0.0
    for i in range(len(vec_a)):
        diff = vec_a[i] - vec_b[i]
        total += weights[i] * (diff * diff)

    return math.sqrt(total)


def validate_weights(weights, expected_length=9):
    """
    Guard function — call before any search to catch bad weights early.

    Rules:
      1. Must have exactly `expected_length` values (default 9).
         This matches VECTOR_LENGTH from data_encoder.py.
      2. All values must be non-negative (negative weights are meaningless
         in distance — they would reward dissimilarity).
      3. At least one must be non-zero (all-zero weights would make every
         profile equidistant, producing meaningless results).

    Args:
        weights:         List of floats to validate.
        expected_length: How many weights are required (default 9).

    Returns:
        True if all checks pass.

    Raises:
        ValueError: with a descriptive message on any violation.
    """
    if len(weights) != expected_length:
        raise ValueError(
            f"Expected {expected_length} weights, got {len(weights)}")

    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")

    if all(w == 0 for w in weights):
        raise ValueError("At least one weight must be non-zero.")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS — run with: python distance.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running distance.py self-tests...\n")

    w = [1.0] * 9
    v = [0.5] * 9

    # Test 1: Distance from a vector to itself is always 0
    assert weighted_distance(v, v, w) == 0.0
    print("  ✅ Test 1 PASSED: Self-distance is 0.0")

    # Test 2: Distance is symmetric — d(A,B) == d(B,A)
    a = [0.1] + [0.0] * 8
    b = [0.9] + [0.0] * 8
    assert abs(weighted_distance(a, b, w) - weighted_distance(b, a, w)) < 1e-9
    print("  ✅ Test 2 PASSED: Symmetry holds")

    # Test 3: A zero weight completely ignores that dimension
    # Only weight on index 0; vectors differ only at index 0
    w_z = [0.0] * 9
    w_z[0] = 1.0
    dist = weighted_distance([0.0] * 9, [1.0] + [0.0] * 8, w_z)
    assert abs(dist - 1.0) < 1e-9
    print("  ✅ Test 3 PASSED: Zero weight ignores dimension")

    # Test 4: All-zero weights should be rejected by validate_weights
    try:
        validate_weights([0.0] * 9)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Test 4 PASSED: All-zero weights rejected")

    # Test 5: Wrong length should be rejected
    try:
        validate_weights([1.0] * 12)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Test 5 PASSED: Wrong length (12) rejected")

    # Test 6: Negative weight should be rejected
    try:
        validate_weights([1.0, -1.0] + [1.0] * 7)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Test 6 PASSED: Negative weight rejected")

    # Test 7: Known value — two vectors differing by 1.0 in all 9 dims
    # with unit weights: sqrt(9 * 1.0²) = 3.0
    d = weighted_distance([0.0] * 9, [1.0] * 9, [1.0] * 9)
    assert abs(d - 3.0) < 1e-9
    print("  ✅ Test 7 PASSED: Known distance sqrt(9) = 3.0")

    # Test 8: Length mismatch raises ValueError
    try:
        weighted_distance([0.0] * 9, [1.0] * 8, [1.0] * 9)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Test 8 PASSED: Length mismatch rejected")

    print("\nAll distance.py tests PASSED ✅")
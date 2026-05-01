# =============================================================================
# FILE: kdtree.py
# OWNER: PERSON 3 — K-D Tree Data Structure + K-NN Search
# PURPOSE: Build a k-d tree index over 9-dimensional encoded profile vectors
#          and perform fast k-nearest-neighbour queries with pruning.
#
# WHY A K-D TREE?
#     The baseline approach (Person 2) computes the distance from the query
#     to every single profile — O(n) per query. With 100,000 profiles, that
#     works but is slow for repeated queries. A k-d tree partitions the
#     vector space into regions so that large portions can be skipped
#     entirely when they cannot contain a closer neighbour than what we
#     already have. This pruning reduces the average query time significantly.
#
# HOW IT WORKS (high level):
#     BUILD: Recursively split the dataset by the dimension with the highest
#            variance, using the median value as the split point. This creates
#            a balanced binary tree where each leaf holds a small cluster of
#            points (default 20).
#
#     SEARCH: Traverse the tree, always visiting the branch closer to the
#             query first. Use a max-heap of size k to track the best
#             candidates. Before visiting the far branch, check if it could
#             possibly contain a better result — if not, prune it entirely.
#
# DIMENSION CHARACTERISTICS (affects tree quality):
#     Indices 0–2 (age, income, hours): continuous [0,1] — excellent splits
#     Index 3 (degree): 4 distinct values {0, 0.333, 0.667, 1.0} — good splits
#     Indices 4–8 (domain one-hot): binary {0, 1} — poor splits, low variance
#
#     The variance-based split selection naturally favours indices 0–3 over
#     4–8. This is correct: splitting on a dimension with only two values
#     (0 or 1) creates unbalanced partitions with little discriminative power.
#     Label-encoded degree at index 3 provides a useful split dimension that
#     would not exist under one-hot encoding (where each of the 4 degree dims
#     would have even lower variance individually).
#
# STDLIB ONLY — no external packages. MaxHeap is built from scratch (no heapq).
# =============================================================================

import time
import math

from distance import weighted_distance, validate_weights


# ─────────────────────────────────────────────────────────────────────────────
# MAX-HEAP — Built from scratch for k-NN candidate tracking
# ─────────────────────────────────────────────────────────────────────────────

class MaxHeap:
    """
    A max-heap storing (distance, profile_id) pairs.

    heap[0] is always the pair with the LARGEST distance — i.e. the worst
    current candidate among our top-k results.

    WHY MAX-HEAP FOR K-NN?
        We want the k SMALLEST distances. A max-heap of fixed size k lets us:
        - Check in O(1) whether a new candidate beats our worst (peek the root)
        - Replace the worst with a better candidate in O(log k) (pop + push)
        - Prune entire subtrees: if the closest possible point in a subtree
          is farther than heap[0], nothing in that subtree can improve our
          results — skip it entirely

        A min-heap would not help here because we need quick access to the
        LARGEST distance (the one we might want to evict), not the smallest.

    WHY NOT heapq?
        Python's heapq is a min-heap. We need max-heap behaviour. Rather
        than using the common hack of negating distances (which is fragile
        and confusing), we implement a proper max-heap from scratch.
        This also satisfies the assignment requirement to build the data
        structure ourselves.

    HEAP INVARIANT:
        For every node at index i, its value is >= both children:
            heap[i] >= heap[2i+1]  (left child)
            heap[i] >= heap[2i+2]  (right child)
    """

    def __init__(self, max_size):
        self.heap = []          # List of (distance, profile_id) tuples
        self.max_size = max_size  # Maximum number of candidates to keep (= k)

    def push(self, dist, pid):
        """
        Add a new (distance, profile_id) pair and restore the heap invariant.

        The new element is appended to the end (bottom of the tree), then
        "sifted up" by swapping with its parent until the max-heap property
        is restored. This is O(log k) where k is the heap size.
        """
        self.heap.append((dist, pid))
        self._sift_up(len(self.heap) - 1)

    def pop_max(self):
        """
        Remove and return the (distance, pid) pair with the LARGEST distance.

        The root (index 0) holds the max. We replace it with the last element,
        then sift that element down to restore the invariant. This is O(log k).

        Raises:
            IndexError: if the heap is empty.
        """
        if len(self.heap) == 0:
            raise IndexError("Pop from empty heap")
        if len(self.heap) == 1:
            return self.heap.pop()
        # Swap root with last element, pop the old root, sift down the new root
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return root

    def peek_max_dist(self):
        """
        Return the largest distance in the heap without removing it — O(1).

        If the heap is empty, return infinity so that any real distance
        will be considered "better" and trigger insertion.
        """
        return self.heap[0][0] if self.heap else float('inf')

    def size(self):
        """Current number of elements in the heap."""
        return len(self.heap)

    def is_full(self):
        """True if the heap has reached its maximum capacity (k candidates)."""
        return len(self.heap) >= self.max_size

    def to_sorted_list(self):
        """
        Return all items sorted ascending by distance.

        This is called once after the search completes to produce the final
        result list. The sort is O(k log k) which is negligible since k <= 20.
        """
        items = sorted(self.heap, key=lambda x: x[0])
        return [{"profile_id": pid, "distance": round(d, 6)}
                for d, pid in items]

    # ── Internal heap operations ─────────────────────────────────────────

    def _sift_up(self, i):
        """
        Move element at index i upward while it is larger than its parent.

        This restores the max-heap invariant after inserting at the bottom.
        At each step, if the child is larger than its parent, swap them
        and continue from the parent's position.
        """
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[parent][0] < self.heap[i][0]:
                self.heap[parent], self.heap[i] = self.heap[i], self.heap[parent]
                i = parent
            else:
                break  # Invariant restored

    def _sift_down(self, i):
        """
        Move element at index i downward while it is smaller than a child.

        This restores the max-heap invariant after replacing the root.
        At each step, find the largest among the node and its two children.
        If the node is not the largest, swap it with the largest child
        and continue from that child's position.
        """
        n = len(self.heap)
        while True:
            largest = i
            left  = 2 * i + 1
            right = 2 * i + 2

            if left < n and self.heap[left][0] > self.heap[largest][0]:
                largest = left
            if right < n and self.heap[right][0] > self.heap[largest][0]:
                largest = right

            if largest == i:
                break  # Invariant restored

            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            i = largest


# ─────────────────────────────────────────────────────────────────────────────
# K-D TREE NODE
# ─────────────────────────────────────────────────────────────────────────────

class KDNode:
    """
    A single node in the k-d tree.

    There are two types:
      - Internal node: stores split_dim and split_val, with left/right children.
        Points with value < split_val go left; points >= split_val go right.
      - Leaf node: stores the actual data points (is_leaf=True).
        The search examines every point in a leaf via brute-force.

    Storing data only in leaves (not internal nodes) simplifies the build
    and search logic. The leaf_size parameter controls how many points
    each leaf holds — a larger leaf_size means fewer tree levels but more
    brute-force work per leaf.
    """
    def __init__(self):
        self.split_dim = None    # int (0–8): which dimension this node splits on
        self.split_val = None    # float: the median value used as the split threshold
        self.left      = None    # KDNode: subtree for points below split_val
        self.right     = None    # KDNode: subtree for points at or above split_val
        self.is_leaf   = False   # True if this is a leaf node holding data
        self.points    = []      # List of (pid, vector) — populated only in leaves


# ─────────────────────────────────────────────────────────────────────────────
# TREE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _choose_split_dim(points):
    """
    Choose the dimension with the highest variance for splitting.

    WHY VARIANCE-BASED SELECTION (not round-robin)?
        The naive approach cycles through dimensions: depth % 9.
        But with our 9-dim vectors, indices 4–8 are one-hot encoded
        domain values — mostly 0s with about 20% being 1. Their variance
        is very low (~0.16), so splitting on them barely separates the data.

        Variance-based selection naturally prefers dimensions 0–3 (age,
        income, hours, degree) which have continuous spread across [0, 1].
        This produces tighter, more balanced partitions and better pruning
        during search.

        Degree at index 3 has 4 distinct values {0, 0.333, 0.667, 1.0},
        giving it decent variance (~0.1). Under one-hot encoding, each of
        the 4 degree dims would have even lower variance individually —
        another benefit of Person 1's label encoding decision.

    Args:
        points: List of (pid, vector) tuples.

    Returns:
        Integer index (0–8) of the dimension with the highest variance.
    """
    num_dims = len(points[0][1])  # = 9
    best_dim = 0
    best_var = -1.0

    for d in range(num_dims):
        # Extract all values in this dimension
        vals = [p[1][d] for p in points]
        # Compute variance: E[(X - μ)²]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        if var > best_var:
            best_var = var
            best_dim = d

    return best_dim


def build_kdtree(points, depth=0, leaf_size=20):
    """
    Recursively build the k-d tree from a list of (pid, vector) tuples.

    ALGORITHM:
        1. BASE CASE: If the number of points <= leaf_size, create a leaf
           node and store all points directly. The search will brute-force
           these few points, which is fast for small groups.

        2. SPLIT DIMENSION: Choose the dimension with the highest variance
           using _choose_split_dim(). This ensures we split where the data
           is most spread out, creating the tightest partitions.

        3. MEDIAN SPLIT: Sort the points by the chosen dimension, take the
           middle element's value as the split threshold. Points below the
           median go left, points at or above go right. Using the median
           keeps the tree roughly balanced (each side gets ~half the points).

        4. RECURSE: Build left and right subtrees from the two halves.

    COMPLEXITY:
        Each level does O(n log n) work for sorting. With O(log n) levels
        (balanced tree), total build time is O(n log²n). In practice, the
        variance computation adds a constant factor of d=9 per level.

    Args:
        points:    List of (pid, vector) tuples to partition.
        depth:     Current recursion depth (for debugging, not used for splitting).
        leaf_size: Maximum points in a leaf before splitting (default 20).

    Returns:
        Root KDNode of the constructed subtree.
    """
    node = KDNode()

    # Base case: few enough points to store directly in a leaf
    if len(points) <= leaf_size:
        node.is_leaf = True
        node.points = points
        return node

    # Choose the dimension that separates the data best
    split_dim = _choose_split_dim(points)

    # Sort by the chosen dimension and split at the median
    sorted_pts = sorted(points, key=lambda p: p[1][split_dim])
    mid = len(sorted_pts) // 2

    node.split_dim = split_dim
    node.split_val = sorted_pts[mid][1][split_dim]

    # Recurse on each half
    node.left  = build_kdtree(sorted_pts[:mid], depth + 1, leaf_size)
    node.right = build_kdtree(sorted_pts[mid:], depth + 1, leaf_size)

    return node


def build_index(encoded_dataset, leaf_size=20):
    """
    Public wrapper for tree construction. Called once at startup in main.py.

    This is separate from build_kdtree() so that main.py has a clean API
    and we can add timing/logging without cluttering the recursive logic.

    Args:
        encoded_dataset: List of (profile_id, vector) tuples from encode_dataset().
        leaf_size:       Max points per leaf (default 20).

    Returns:
        Root KDNode of the built tree.
    """
    print(f"Building k-d tree (n={len(encoded_dataset):,})...")
    start = time.time()
    tree = build_kdtree(encoded_dataset, depth=0, leaf_size=leaf_size)
    elapsed = time.time() - start
    print(f"Tree built in {elapsed:.2f}s")
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# K-NN SEARCH WITH PRUNING — The core algorithm
# ─────────────────────────────────────────────────────────────────────────────

def _kdtree_knn_recursive(node, query_vec, weights, k, heap):
    """
    Recursively search the k-d tree for the k nearest neighbours.

    This is the most critical function in the module. It uses a max-heap
    of size k to track the best candidates found so far, and prunes
    entire subtrees that cannot contain better results.

    LEAF NODE LOGIC:
        Brute-force check every point in the leaf. For each:
        - If the heap is not full yet, add it (we need more candidates).
        - If it is closer than the current worst (heap root), evict the
          worst and add the new candidate.

    INTERNAL NODE LOGIC:
        1. Determine which branch is "near" (same side as the query) and
           which is "far" (opposite side of the splitting hyperplane).
        2. ALWAYS recurse into the near branch — it is more likely to
           contain close neighbours.
        3. PRUNING CHECK: Before visiting the far branch, compute the
           minimum possible distance from the query to any point on the
           far side of the hyperplane:

               plane_dist = sqrt(w[split_dim]) × |query[split_dim] - split_val|

           This is the weighted distance from the query to the splitting
           plane along the split dimension alone. Any point on the far side
           must be at LEAST this far away.

           Visit the far branch ONLY IF:
               - The heap is not full yet (we still need more candidates), OR
               - plane_dist < current worst distance in the heap

           If neither condition holds, the entire far subtree is pruned.
           This is what makes the k-d tree faster than brute-force: on
           average, it skips a large fraction of the tree.

    Args:
        node:      Current KDNode being examined.
        query_vec: The 9-dim query vector.
        weights:   Per-dimension weight list (9 floats).
        k:         Number of neighbours to find.
        heap:      MaxHeap instance tracking the best k candidates so far.
    """
    # ── LEAF NODE: check every point directly ────────────────────────────
    if node.is_leaf:
        for pid, vec in node.points:
            d = weighted_distance(query_vec, vec, weights)
            if not heap.is_full():
                # Still collecting initial candidates — add unconditionally
                heap.push(d, pid)
            elif d < heap.peek_max_dist():
                # Found a closer point than our current worst — swap it in
                heap.pop_max()
                heap.push(d, pid)
        return

    # ── INTERNAL NODE: decide which branch to visit ──────────────────────
    split_dim = node.split_dim
    split_val = node.split_val

    # The "near" branch is on the same side of the hyperplane as the query.
    # We visit it first because it is more likely to contain close points.
    if query_vec[split_dim] < split_val:
        near_branch = node.left
        far_branch  = node.right
    else:
        near_branch = node.right
        far_branch  = node.left

    # Always recurse into the near branch
    _kdtree_knn_recursive(near_branch, query_vec, weights, k, heap)

    # ── PRUNING CHECK ────────────────────────────────────────────────────
    # Compute the minimum weighted distance from the query to the splitting
    # hyperplane. This is the absolute difference along the split dimension,
    # scaled by the square root of that dimension's weight.
    #
    # Why sqrt(weight)? The weighted distance formula is:
    #   d = sqrt(Σ w_i × (a_i - b_i)²)
    # The contribution of a single dimension i is sqrt(w_i) × |a_i - b_i|
    # when considered in isolation (all other differences are zero).
    plane_dist = abs(query_vec[split_dim] - split_val)
    weighted_plane_dist = (weights[split_dim] ** 0.5) * plane_dist

    # Visit the far branch only if it might contain a better candidate:
    #   - If we don't have k candidates yet, we MUST explore further
    #   - If the plane distance is less than our worst, there might be
    #     closer points on the far side
    should_visit_far = (
        not heap.is_full()
        or weighted_plane_dist < heap.peek_max_dist()
    )

    if should_visit_far:
        _kdtree_knn_recursive(far_branch, query_vec, weights, k, heap)


def search(tree, query_vec, weights, k):
    """
    Public search function. Called from main.py for each query.

    Wraps the recursive search with input validation, timing, and result
    formatting. The return format matches baseline_knn() exactly so that
    main.py can compare results directly.

    Args:
        tree:      Root KDNode from build_index().
        query_vec: List of 9 floats — the encoded query profile.
        weights:   List of 9 non-negative floats — per-dimension weights.
        k:         Number of nearest neighbours to return (1–20).

    Returns:
        Tuple of (results, elapsed_seconds) where results is a list of
        {"profile_id": int, "distance": float} dicts, sorted by distance
        ascending — identical format to baseline_knn().
    """
    validate_weights(weights)
    if not (1 <= k <= 20):
        raise ValueError(f"k must be 1–20, got {k}")

    start = time.time()

    # Create a max-heap with capacity k to track the best candidates
    heap = MaxHeap(max_size=k)

    # Run the recursive search — this populates the heap
    _kdtree_knn_recursive(tree, query_vec, weights, k, heap)

    # Convert the heap into a sorted result list
    results = heap.to_sorted_list()

    elapsed = time.time() - start
    return results, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION AGAINST BASELINE — Correctness proof
# ─────────────────────────────────────────────────────────────────────────────

def validate_against_baseline(tree, encoded_dataset, weights, k,
                              n_queries=50):
    """
    Run n_queries random queries and verify the k-d tree returns the same
    results as the brute-force baseline.

    This is the ultimate correctness check. The baseline is guaranteed to
    find the exact k nearest neighbours (it checks everything). If the
    k-d tree returns different results, it means the pruning logic has a
    bug — either pruning too aggressively (skipping subtrees that contain
    closer points) or not pruning at all (which would still be correct
    but slow).

    We use random profiles and random weights to stress-test diverse
    query patterns, not just the easy cases.

    Args:
        tree:            Root KDNode.
        encoded_dataset: List of (pid, vector) tuples.
        weights:         Default weights (overridden with random per query).
        k:               Number of neighbours per query.
        n_queries:       How many random queries to test (default 50).

    Returns:
        True if all queries match, False if any mismatch was found.
    """
    import random
    from baseline import baseline_knn
    from data_encoder import DEGREES, DOMAINS, encode_profile

    print(f"\nValidating against baseline ({n_queries} queries)...")

    all_ok = True
    for i in range(n_queries):
        # Generate a random query profile
        qp = {
            "age":                 random.randint(18, 70),
            "income":              random.randint(5, 100),
            "highest_degree":      random.choice(DEGREES),
            "self_learning_hours": round(random.uniform(0, 4), 2),
            "favourite_domain":    random.choice(DOMAINS),
        }
        qvec = encode_profile(qp)

        # Random weights — ensures we test diverse weight configurations
        w = [round(random.uniform(0.1, 3.0), 2) for _ in range(9)]

        # Run both search methods
        kd_res, _ = search(tree, qvec, w, k)
        bl_res, _ = baseline_knn(qvec, encoded_dataset, w, k)

        # Compare the profile IDs returned by each method
        kd_ids = [r["profile_id"] for r in kd_res]
        bl_ids = [r["profile_id"] for r in bl_res]

        if kd_ids != bl_ids:
            print(f"  ❌ Query {i + 1} FAILED")
            print(f"     Profile: {qp}")
            print(f"     Weights: {w}")
            print(f"     Baseline: {bl_ids}")
            print(f"     KD-Tree:  {kd_ids}")
            all_ok = False

    if all_ok:
        print(f"  ✅ All {n_queries} queries matched baseline perfectly")
    else:
        print(f"  ⚠️  FAILURES detected — check pruning logic in "
              f"_kdtree_knn_recursive()")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS — run with: python kdtree.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_encoder import generate_dataset, encode_dataset, encode_profile

    print("Running kdtree.py self-tests...\n")

    # Build a tree from 5,000 profiles
    ds     = generate_dataset(n=5000, seed=42)
    enc_ds = encode_dataset(ds)
    tree   = build_index(enc_ds, leaf_size=20)

    # Test query
    qvec = encode_profile({
        "age": 28, "income": 50, "highest_degree": "Master",
        "self_learning_hours": 1.5, "favourite_domain": "Data Science"
    })

    # Test 1: Basic search returns correct number of results
    results, elapsed = search(tree, qvec, [1.0] * 9, k=5)
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    print(f"  ✅ Test 1 PASSED: k=5 returns 5 results ({elapsed*1000:.2f}ms)")

    # Test 2: Results are sorted by distance ascending
    for i in range(len(results) - 1):
        assert results[i]["distance"] <= results[i + 1]["distance"], \
            "Results not sorted by distance"
    print("  ✅ Test 2 PASSED: Results sorted by distance")

    # Test 3: All distances are non-negative
    for r in results:
        assert r["distance"] >= 0, f"Negative distance: {r['distance']}"
    print("  ✅ Test 3 PASSED: All distances non-negative")

    # Test 4: Invalid k is rejected
    try:
        search(tree, qvec, [1.0] * 9, k=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Test 4 PASSED: k=0 rejected")

    # Test 5: Print top-5 results
    print(f"\n  Top-5 nearest neighbours (k-d tree, {elapsed * 1000:.2f}ms):")
    for r in results:
        print(f"    ID={r['profile_id']:<6} distance={r['distance']}")

    # Test 6: MaxHeap unit tests
    print("\n  ── MaxHeap unit tests ──")
    h = MaxHeap(max_size=3)
    assert h.size() == 0 and not h.is_full()
    h.push(5.0, 100)
    h.push(2.0, 200)
    h.push(8.0, 300)
    assert h.is_full()
    assert h.peek_max_dist() == 8.0
    print("  ✅ Test 6a PASSED: MaxHeap push/peek correct")

    removed = h.pop_max()
    assert removed == (8.0, 300)
    assert h.peek_max_dist() == 5.0
    print("  ✅ Test 6b PASSED: MaxHeap pop_max correct")

    # Push a smaller value — should be accepted
    h.push(1.0, 400)
    assert h.is_full()
    sorted_items = h.to_sorted_list()
    assert sorted_items[0]["distance"] <= sorted_items[-1]["distance"]
    print("  ✅ Test 6c PASSED: MaxHeap to_sorted_list correct")

    # Test 7: Validate against baseline with 50 random queries
    validate_against_baseline(tree, enc_ds, [1.0] * 9, k=5, n_queries=50)

    print("\nAll kdtree.py tests PASSED ✅")
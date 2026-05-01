"""
test_all.py — Full integration test suite for the Scalable Similarity Search system.
Covers all modules: data_encoder, distance, baseline, kdtree.
Sections A–K as specified in the master build instructions.

Run:  python test_all.py
"""

import random
import os
import math

# ─── Imports from project modules ─────────────────────────────────────────

from data_encoder import (
    DEGREES, DOMAINS, VECTOR_LENGTH, DEGREE_ORDER, DEGREE_MAX_LABEL,
    AGE_MIN, AGE_MAX, INCOME_MIN, INCOME_MAX, HOURS_MIN, HOURS_MAX,
    generate_dataset, save_dataset, load_dataset,
    get_or_create_dataset, validate_profile, encode_profile,
    encode_dataset, get_dataset_stats, decode_vector
)
from distance import weighted_distance, validate_weights
from baseline import baseline_knn
from kdtree import build_index, search as kdtree_search, validate_against_baseline


# ─── Test infrastructure ──────────────────────────────────────────────────

passed = 0
failed = 0


def check(label, condition):
    """Record a single test result."""
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  ❌ FAIL: {label}")


def section(title):
    print(f"\n── {title} ──")


# ═══════════════════════════════════════════════════════════════════════════
# A. Constants
# ═══════════════════════════════════════════════════════════════════════════

section("A. Constants")
check("VECTOR_LENGTH == 9", VECTOR_LENGTH == 9)
check("DEGREE_MAX_LABEL == 3", DEGREE_MAX_LABEL == 3)
check("DEGREES has 4 items", len(DEGREES) == 4)
check("DEGREES order", DEGREES == ["High School", "Bachelor", "Master", "PhD"])
check("DOMAINS has 5 items", len(DOMAINS) == 5)
check("DOMAINS order",
      DOMAINS == ["AI", "Software Engineering", "Data Science",
                  "Cybersecurity", "Business Analytics"])
check("DEGREE_ORDER maps correctly",
      DEGREE_ORDER == {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3})


# ═══════════════════════════════════════════════════════════════════════════
# B. Profile generation
# ═══════════════════════════════════════════════════════════════════════════

section("B. Profile generation")

ds500 = generate_dataset(n=500, seed=0)
check("generate_dataset returns correct count", len(ds500) == 500)
check("Sequential IDs from 0",
      all(ds500[i]["id"] == i for i in range(500)))

for p in ds500:
    if not (AGE_MIN <= p["age"] <= AGE_MAX):
        check(f"age in range for id={p['id']}", False)
        break
    if not (INCOME_MIN <= p["income"] <= INCOME_MAX):
        check(f"income in range for id={p['id']}", False)
        break
    if p["highest_degree"] not in DEGREES:
        check(f"degree valid for id={p['id']}", False)
        break
    if not (HOURS_MIN <= p["self_learning_hours"] <= HOURS_MAX):
        check(f"hours in range for id={p['id']}", False)
        break
    if p["favourite_domain"] not in DOMAINS:
        check(f"domain valid for id={p['id']}", False)
        break
else:
    check("All values in valid ranges", True)

# Same seed = same data
ds500_copy = generate_dataset(n=500, seed=0)
check("Same seed = same data",
      all(ds500[i]["age"] == ds500_copy[i]["age"] for i in range(500)))

# Different seed = different data
ds500_diff = generate_dataset(n=500, seed=99)
check("Different seed = different data",
      any(ds500[i]["age"] != ds500_diff[i]["age"] for i in range(500)))


# ═══════════════════════════════════════════════════════════════════════════
# C. CSV save/load round-trip
# ═══════════════════════════════════════════════════════════════════════════

section("C. CSV save/load round-trip")

test_csv = "__test_roundtrip.csv"
save_dataset(ds500[:50], test_csv)
check("CSV file created", os.path.exists(test_csv))
check("CSV not empty", os.path.getsize(test_csv) > 0)

loaded = load_dataset(test_csv)
check("Loaded count matches", len(loaded) == 50)
check("id is int after load", isinstance(loaded[0]["id"], int))
check("age is int after load", isinstance(loaded[0]["age"], int))
check("income is int after load", isinstance(loaded[0]["income"], int))
check("hours is float after load",
      isinstance(loaded[0]["self_learning_hours"], float))
check("degree is str after load",
      isinstance(loaded[0]["highest_degree"], str))
check("Values match original",
      loaded[0]["age"] == ds500[0]["age"] and
      loaded[0]["income"] == ds500[0]["income"])

# Cleanup
os.remove(test_csv)


# ═══════════════════════════════════════════════════════════════════════════
# D. validate_profile
# ═══════════════════════════════════════════════════════════════════════════

section("D. validate_profile")

valid_p = {"age": 30, "income": 50, "highest_degree": "Bachelor",
           "self_learning_hours": 2.0, "favourite_domain": "AI"}
check("Valid profile returns True", validate_profile(valid_p) is True)

# Missing key
try:
    validate_profile({"age": 30, "income": 50})
    check("Missing key raises ValueError", False)
except ValueError:
    check("Missing key raises ValueError", True)

# Out of range age
for bad_age in [17, 71]:
    try:
        validate_profile({**valid_p, "age": bad_age})
        check(f"age={bad_age} raises ValueError", False)
    except ValueError:
        check(f"age={bad_age} raises ValueError", True)

# Out of range income
for bad_inc in [4, 101]:
    try:
        validate_profile({**valid_p, "income": bad_inc})
        check(f"income={bad_inc} raises ValueError", False)
    except ValueError:
        check(f"income={bad_inc} raises ValueError", True)

# Out of range hours
for bad_hrs in [-0.1, 4.1]:
    try:
        validate_profile({**valid_p, "self_learning_hours": bad_hrs})
        check(f"hours={bad_hrs} raises ValueError", False)
    except ValueError:
        check(f"hours={bad_hrs} raises ValueError", True)

# Invalid degree/domain
try:
    validate_profile({**valid_p, "highest_degree": "Diploma"})
    check("Invalid degree raises ValueError", False)
except ValueError:
    check("Invalid degree raises ValueError", True)

try:
    validate_profile({**valid_p, "favourite_domain": "Biology"})
    check("Invalid domain raises ValueError", False)
except ValueError:
    check("Invalid domain raises ValueError", True)

# Wrong case
try:
    validate_profile({**valid_p, "highest_degree": "bachelor"})
    check("Wrong case degree raises ValueError", False)
except ValueError:
    check("Wrong case degree raises ValueError", True)

# Boundary values are valid
boundary_p = {"age": 18, "income": 5, "highest_degree": "High School",
              "self_learning_hours": 0.0, "favourite_domain": "AI"}
check("Boundary min valid", validate_profile(boundary_p) is True)

boundary_max = {"age": 70, "income": 100, "highest_degree": "PhD",
                "self_learning_hours": 4.0, "favourite_domain": "Business Analytics"}
check("Boundary max valid", validate_profile(boundary_max) is True)


# ═══════════════════════════════════════════════════════════════════════════
# E. encode_profile — vector shape
# ═══════════════════════════════════════════════════════════════════════════

section("E. encode_profile")

v = encode_profile(valid_p)
check("Vector length == 9", len(v) == 9)
check("All values in [0, 1]", all(0.0 <= x <= 1.0 for x in v))

# Boundary profiles
v_min = encode_profile(boundary_p)
check("Min age → 0.0", v_min[0] == 0.0)
check("Min income → 0.0", v_min[1] == 0.0)
check("Min hours → 0.0", v_min[2] == 0.0)

v_max = encode_profile(boundary_max)
check("Max age → 1.0", v_max[0] == 1.0)
check("Max income → 1.0", v_max[1] == 1.0)
check("Max hours → 1.0", v_max[2] == 1.0)

# Degree label encoding
for deg, expected in [("High School", 0.0), ("Bachelor", 1/3),
                      ("Master", 2/3), ("PhD", 1.0)]:
    vd = encode_profile({**valid_p, "highest_degree": deg})
    check(f"Degree {deg} → {expected:.4f}",
          abs(vd[3] - expected) < 1e-9)

# Labels strictly increasing
deg_vals = []
for deg in DEGREES:
    vd = encode_profile({**valid_p, "highest_degree": deg})
    deg_vals.append(vd[3])
check("Degree labels strictly increasing",
      all(deg_vals[i] < deg_vals[i+1] for i in range(3)))

# Domain one-hot
for i, dom in enumerate(DOMAINS):
    vdom = encode_profile({**valid_p, "favourite_domain": dom})
    check(f"Domain {dom} one-hot sums to 1.0",
          abs(sum(vdom[4:9]) - 1.0) < 1e-9)
    check(f"Domain {dom} hot at index {4+i}", vdom[4+i] == 1.0)

# Normalisation spot-check: age=25 → (25-18)/52
v25 = encode_profile({**valid_p, "age": 25})
check("age=25 normalised correctly", abs(v25[0] - 7/52) < 1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# F. encode_dataset
# ═══════════════════════════════════════════════════════════════════════════

section("F. encode_dataset")

enc_small = encode_dataset(ds500[:10])
check("Returns list of tuples", isinstance(enc_small, list))
check("Correct count", len(enc_small) == 10)
check("Tuple is (int, list)", isinstance(enc_small[0], tuple) and
      isinstance(enc_small[0][0], int) and isinstance(enc_small[0][1], list))
check("IDs match", enc_small[0][0] == ds500[0]["id"])


# ═══════════════════════════════════════════════════════════════════════════
# G. decode_vector
# ═══════════════════════════════════════════════════════════════════════════

section("G. decode_vector")

# Degree round-trip
for deg in DEGREES:
    p_test = {**valid_p, "highest_degree": deg}
    v_test = encode_profile(p_test)
    d_test = decode_vector(v_test)
    check(f"Degree {deg} round-trips", d_test["highest_degree"] == deg)

# Domain round-trip
for dom in DOMAINS:
    p_test = {**valid_p, "favourite_domain": dom}
    v_test = encode_profile(p_test)
    d_test = decode_vector(v_test)
    check(f"Domain {dom} round-trips", d_test["favourite_domain"] == dom)

# Numeric round-trip within ±1
p_num = {"age": 35, "income": 60, "highest_degree": "Master",
         "self_learning_hours": 2.5, "favourite_domain": "Data Science"}
v_num = encode_profile(p_num)
d_num = decode_vector(v_num)
check("age round-trip ±1", abs(d_num["age"] - 35) <= 1)
check("income round-trip ±1", abs(d_num["income"] - 60) <= 1)
check("hours round-trip ±0.05", abs(d_num["self_learning_hours"] - 2.5) <= 0.05)

# Wrong length raises ValueError
for bad_len in [8, 12]:
    try:
        decode_vector([0.5] * bad_len)
        check(f"decode len={bad_len} raises ValueError", False)
    except ValueError:
        check(f"decode len={bad_len} raises ValueError", True)


# ═══════════════════════════════════════════════════════════════════════════
# H. get_dataset_stats
# ═══════════════════════════════════════════════════════════════════════════

section("H. get_dataset_stats")

stats = get_dataset_stats(ds500)
check("stats['n'] == 500", stats["n"] == 500)
check("degree_counts sum to n",
      sum(stats["degree_counts"].values()) == 500)
check("domain_counts sum to n",
      sum(stats["domain_counts"].values()) == 500)
check("All degree categories present",
      all(d in stats["degree_counts"] for d in DEGREES))
check("All domain categories present",
      all(d in stats["domain_counts"] for d in DOMAINS))


# ═══════════════════════════════════════════════════════════════════════════
# I. Distance function
# ═══════════════════════════════════════════════════════════════════════════

section("I. Distance function")

w = [1.0] * 9
va = [0.5] * 9
check("Self-distance = 0", weighted_distance(va, va, w) == 0.0)

a = [0.1] + [0.0] * 8
b = [0.9] + [0.0] * 8
check("Symmetry", abs(weighted_distance(a, b, w) -
                       weighted_distance(b, a, w)) < 1e-12)

# Zero weight ignores dimension
wz = [0.0] * 9
wz[0] = 1.0
check("Zero weight ignores dim",
      abs(weighted_distance([0.0]*9, [0.0, 0.5]+[0.0]*7, wz) - 0.0) < 1e-12)

# Wrong length raises ValueError
try:
    weighted_distance([0.5]*9, [0.5]*8, w)
    check("Wrong length raises ValueError", False)
except ValueError:
    check("Wrong length raises ValueError", True)


# ═══════════════════════════════════════════════════════════════════════════
# J. Baseline search
# ═══════════════════════════════════════════════════════════════════════════

section("J. Baseline search")

ds1k = generate_dataset(n=1000, seed=42)
enc1k = encode_dataset(ds1k)
qv = encode_profile(valid_p)
res, t = baseline_knn(qv, enc1k, [1.0]*9, 5)

check("Returns exactly k results", len(res) == 5)
check("Sorted ascending by distance",
      all(res[i]["distance"] <= res[i+1]["distance"] for i in range(4)))
check("Has profile_id key", all("profile_id" in r for r in res))
check("Has distance key", all("distance" in r for r in res))
check("profile_id is int", isinstance(res[0]["profile_id"], int))
check("distance is float", isinstance(res[0]["distance"], float))


# ═══════════════════════════════════════════════════════════════════════════
# K. K-D tree vs baseline
# ═══════════════════════════════════════════════════════════════════════════

section("K. K-D tree vs baseline")

ds5k = generate_dataset(n=5000, seed=42)
enc5k = encode_dataset(ds5k)
tree = build_index(enc5k, leaf_size=20)
check("build_index completes", tree is not None)

kd_res, kd_t = kdtree_search(tree, qv, [1.0]*9, 5)
check("search() returns same format",
      len(kd_res) == 5 and "profile_id" in kd_res[0] and
      "distance" in kd_res[0])

# 50 random queries validation
print("  Running 50 random query validations...")
random.seed(12345)
all_match = True
for i in range(50):
    rp = {
        "age": random.randint(18, 70),
        "income": random.randint(5, 100),
        "highest_degree": random.choice(DEGREES),
        "self_learning_hours": round(random.uniform(0.0, 4.0), 2),
        "favourite_domain": random.choice(DOMAINS),
    }
    rqv = encode_profile(rp)
    rw = [round(random.uniform(0.1, 5.0), 2) for _ in range(9)]
    rk = random.choice([1, 3, 5, 10])

    bl_r, _ = baseline_knn(rqv, enc5k, rw, rk)
    kd_r, _ = kdtree_search(tree, rqv, rw, rk)

    bl_ids = [r["profile_id"] for r in bl_r]
    kd_ids = [r["profile_id"] for r in kd_r]
    if bl_ids != kd_ids:
        all_match = False
        print(f"    ⚠️  Query {i}: BL={bl_ids} KD={kd_ids}")

check("50 random queries: KD-tree matches baseline", all_match)


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print(f"  TOTAL: {passed} passed, {failed} failed")
if failed == 0:
    print("  ✅ ALL TESTS PASSED")
else:
    print("  ❌ FIX FAILURES")
print("=" * 60)
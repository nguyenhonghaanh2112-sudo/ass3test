# =============================================================================
# FILE: test_encoder.py
# OWNER: PERSON 1
# RUN:   python test_encoder.py
# =============================================================================

import sys, os

try:
    from data_encoder import (
        DEGREES, DOMAINS, VECTOR_LENGTH, DEGREE_ORDER,
        AGE_MIN, AGE_MAX, INCOME_MIN, INCOME_MAX, HOURS_MAX, DEGREE_MAX_LABEL,
        generate_dataset, encode_profile, encode_dataset,
        save_dataset, load_dataset, get_or_create_dataset,
        validate_profile, get_dataset_stats, decode_vector,
    )
    print("✅ Import OK\n")
except ImportError as e:
    print(f"❌ IMPORT FAILED: {e}")
    sys.exit(1)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅  {name}")
        PASS += 1
    else:
        print(f"  ❌  {name}" + (f"  [{detail}]" if detail else ""))
        FAIL += 1

# ──────────────────────────────────────────────────────────────
print("─"*60)
print("A. Constants")
print("─"*60)

check("DEGREES length == 4",              len(DEGREES) == 4)
check("DOMAINS length == 5",              len(DOMAINS) == 5)
check("VECTOR_LENGTH == 9",               VECTOR_LENGTH == 9)
check("DEGREE_MAX_LABEL == 3",            DEGREE_MAX_LABEL == 3)
check("DEGREES[0] == 'High School'",      DEGREES[0] == "High School")
check("DEGREES[3] == 'PhD'",              DEGREES[3] == "PhD")
check("DOMAINS[0] == 'AI'",              DOMAINS[0] == "AI")
check("DOMAINS[4] == 'Business Analytics'", DOMAINS[4] == "Business Analytics")
check("DEGREE_ORDER High School == 0",   DEGREE_ORDER["High School"] == 0)
check("DEGREE_ORDER Bachelor == 1",      DEGREE_ORDER["Bachelor"] == 1)
check("DEGREE_ORDER Master == 2",        DEGREE_ORDER["Master"] == 2)
check("DEGREE_ORDER PhD == 3",           DEGREE_ORDER["PhD"] == 3)

# ──────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("B. Dataset Generation")
print("─"*60)

ds = generate_dataset(n=300, seed=42)
check("returns list",            isinstance(ds, list))
check("correct count",           len(ds) == 300)
check("first id == 0",           ds[0]["id"] == 0)
check("ids are sequential",      all(ds[i]["id"] == i for i in range(300)))
check("has 6 keys per profile",  all(len(p) == 6 for p in ds))

for p in ds[:50]:
    check(f"age in range  (id={p['id']})", AGE_MIN <= p["age"] <= AGE_MAX)
    check(f"income in range(id={p['id']})", INCOME_MIN <= p["income"] <= INCOME_MAX)
    check(f"hours in range (id={p['id']})", 0.0 <= p["self_learning_hours"] <= HOURS_MAX)
    check(f"degree valid   (id={p['id']})", p["highest_degree"] in DEGREES)
    check(f"domain valid   (id={p['id']})", p["favourite_domain"] in DOMAINS)

ds_a = generate_dataset(n=50, seed=99)
ds_b = generate_dataset(n=50, seed=99)
ds_c = generate_dataset(n=50, seed=100)
check("same seed → same data",      ds_a == ds_b)
check("diff seed → diff data",      ds_a != ds_c)

# ──────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("C. CSV Save / Load Round-Trip")
print("─"*60)

TMP = "_tmp_test.csv"
ds_orig = generate_dataset(n=100, seed=3)
save_dataset(ds_orig, TMP)
check("file created",       os.path.exists(TMP))
check("file not empty",     os.path.getsize(TMP) > 0)

ds_back = load_dataset(TMP)
check("loaded count matches",  len(ds_back) == 100)
check("id is int",             isinstance(ds_back[0]["id"], int))
check("age is int",            isinstance(ds_back[0]["age"], int))
check("income is int",         isinstance(ds_back[0]["income"], int))
check("hours is float",        isinstance(ds_back[0]["self_learning_hours"], float))
check("degree is str",         isinstance(ds_back[0]["highest_degree"], str))
check("domain is str",         isinstance(ds_back[0]["favourite_domain"], str))

fidelity = all(
    ds_orig[i]["age"]    == ds_back[i]["age"]    and
    ds_orig[i]["income"] == ds_back[i]["income"] and
    ds_orig[i]["highest_degree"] == ds_back[i]["highest_degree"] and
    abs(ds_orig[i]["self_learning_hours"] -
        ds_back[i]["self_learning_hours"]) < 0.001 and
    ds_orig[i]["favourite_domain"] == ds_back[i]["favourite_domain"]
    for i in range(100)
)
check("round-trip fidelity", fidelity)
os.remove(TMP)

# ──────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("D. validate_profile()")
print("─"*60)

GOOD = {"age": 25, "income": 40, "highest_degree": "Master",
        "self_learning_hours": 2.0, "favourite_domain": "AI"}

check("valid profile returns True",  validate_profile(GOOD) == True)

# Missing keys
for key in ["age", "income", "highest_degree", "self_learning_hours",
            "favourite_domain"]:
    try:
        validate_profile({k: v for k, v in GOOD.items() if k != key})
        check(f"missing '{key}' raises ValueError", False, "no exception")
    except ValueError:
        check(f"missing '{key}' raises ValueError", True)

# Age out of range
for bad_age in [17, 71, 0, -1, 200]:
    try:
        validate_profile({**GOOD, "age": bad_age})
        check(f"age={bad_age} raises ValueError", False, "no exception")
    except ValueError:
        check(f"age={bad_age} raises ValueError", True)

check("age=18 boundary valid", validate_profile({**GOOD, "age": 18}) == True)
check("age=70 boundary valid", validate_profile({**GOOD, "age": 70}) == True)

# Income out of range
for bad in [4, 101]:
    try:
        validate_profile({**GOOD, "income": bad})
        check(f"income={bad} raises ValueError", False, "no exception")
    except ValueError:
        check(f"income={bad} raises ValueError", True)

# Degree invalid
for bad_deg in ["bachelor", "MASTER", "Doctorate", "", "highschool"]:
    try:
        validate_profile({**GOOD, "highest_degree": bad_deg})
        check(f"degree='{bad_deg}' raises ValueError", False, "no exception")
    except ValueError:
        check(f"degree='{bad_deg}' raises ValueError", True)

# Domain invalid
try:
    validate_profile({**GOOD, "favourite_domain": "Machine Learning"})
    check("unknown domain raises ValueError", False, "no exception")
except ValueError:
    check("unknown domain raises ValueError", True)

# Hours boundary
check("hours=0.0 boundary valid",
      validate_profile({**GOOD, "self_learning_hours": 0.0}) == True)
check("hours=4.0 boundary valid",
      validate_profile({**GOOD, "self_learning_hours": 4.0}) == True)
try:
    validate_profile({**GOOD, "self_learning_hours": -0.1})
    check("hours=-0.1 raises ValueError", False, "no exception")
except ValueError:
    check("hours=-0.1 raises ValueError", True)

# ──────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("E. encode_profile() — vector shape and values")
print("─"*60)

p_hs = {"age":18,"income":5, "highest_degree":"High School",
        "self_learning_hours":0.0,"favourite_domain":"AI"}
p_phd = {"age":70,"income":100,"highest_degree":"PhD",
         "self_learning_hours":4.0,"favourite_domain":"Business Analytics"}
v_hs  = encode_profile(p_hs)
v_phd = encode_profile(p_phd)

check("vector is list",              isinstance(v_hs, list))
check("vector length == 9",          len(v_hs) == 9)
check("all floats",                  all(isinstance(x, float) for x in v_hs))
check("all in [0,1]",                all(0.0 <= x <= 1.0 for x in v_hs))

# Boundary values — min profile
check("age=18 → index0=0.0",         v_hs[0] == 0.0)
check("income=5 → index1=0.0",       v_hs[1] == 0.0)
check("hours=0.0 → index2=0.0",      v_hs[2] == 0.0)
check("HS → index3=0.0",             v_hs[3] == 0.0)
check("AI → index4=1.0",             v_hs[4] == 1.0)
check("AI → index5-8=0.0",           all(v_hs[i] == 0.0 for i in range(5,9)))

# Boundary values — max profile
check("age=70 → index0=1.0",         v_phd[0] == 1.0)
check("income=100 → index1=1.0",     v_phd[1] == 1.0)
check("hours=4.0 → index2=1.0",      v_phd[2] == 1.0)
check("PhD → index3=1.0",            v_phd[3] == 1.0)
check("BizAnal → index8=1.0",        v_phd[8] == 1.0)
check("BizAnal → index4-7=0.0",      all(v_phd[i]==0.0 for i in range(4,8)))

# Degree label values (the critical new test)
degree_label_map = {
    "High School": 0.0,
    "Bachelor":    1/3,
    "Master":      2/3,
    "PhD":         1.0,
}
for deg, expected in degree_label_map.items():
    v = encode_profile({**GOOD, "highest_degree": deg})
    check(f"degree '{deg}' → {round(expected,4)}",
          abs(v[3] - expected) < 1e-6,
          f"got {v[3]}")

# Label order is strictly increasing (crucial property)
labels = []
for deg in DEGREES:
    v = encode_profile({**GOOD, "highest_degree": deg})
    labels.append(v[3])
check("degree labels strictly increasing",
      all(labels[i] < labels[i+1] for i in range(len(labels)-1)),
      f"labels={labels}")

# Domain one-hot (indices 4–8)
for dom in DOMAINS:
    v = encode_profile({**GOOD, "favourite_domain": dom})
    idx = DOMAINS.index(dom)
    check(f"domain '{dom}' → index {4+idx}=1.0", v[4 + idx] == 1.0)
    check(f"domain '{dom}' → others=0.0",
          all(v[4+j]==0.0 for j in range(len(DOMAINS)) if j != idx))
    check(f"domain '{dom}' → one-hot sum=1.0", abs(sum(v[4:9]) - 1.0) < 1e-9)

# Normalisation formula spot-check
p_mid = {"age":44,"income":52,"highest_degree":"Bachelor",
         "self_learning_hours":2.0,"favourite_domain":"Data Science"}
v_mid = encode_profile(p_mid)
check("age formula:    (44-18)/52",  abs(v_mid[0] - (44-18)/52) < 1e-9)
check("income formula: (52-5)/95",   abs(v_mid[1] - (52-5)/95)  < 1e-9)
check("hours formula:  2.0/4.0",     abs(v_mid[2] - 0.5)        < 1e-9)
check("Bachelor label: 1/3",         abs(v_mid[3] - 1/3)        < 1e-9)
check("DataSci → index6=1.0",        v_mid[6] == 1.0)

# ──────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("F. encode_dataset()")
print("─"*60)

ds_small = generate_dataset(n=30, seed=0)
enc      = encode_dataset(ds_small)
check("returns list",                isinstance(enc, list))
check("same length as dataset",      len(enc) == 30)
check("each item is tuple",          all(isinstance(e, tuple) for e in enc))
check("tuple[0] is int (id)",        isinstance(enc[0][0], int))
check("tuple[1] is list (vec)",      isinstance(enc[0][1], list))
check("vector has 9 dims",           len(enc[0][1]) == 9)
check("ids match dataset",           all(enc[i][0]==ds_small[i]["id"]
                                          for i in range(30)))

# ──────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("G. get_dataset_stats()")
print("─"*60)

ds_stat = generate_dataset(n=1000, seed=42)
stats   = get_dataset_stats(ds_stat)

check("returns dict",                  isinstance(stats, dict))
check("n == 1000",                     stats["n"] == 1000)
check("age_min >= 18",                 stats["age_min"] >= 18)
check("age_max <= 70",                 stats["age_max"] <= 70)
check("income_min >= 5",               stats["income_min"] >= 5)
check("income_max <= 100",             stats["income_max"] <= 100)
check("hours_min >= 0.0",              stats["hours_min"] >= 0.0)
check("hours_max <= 4.0",              stats["hours_max"] <= 4.0)
check("degree_counts is dict",         isinstance(stats["degree_counts"], dict))
check("all 4 degrees present",
      all(d in stats["degree_counts"] for d in DEGREES))
check("degree counts sum to n",
      sum(stats["degree_counts"].values()) == 1000)
check("domain_counts is dict",         isinstance(stats["domain_counts"], dict))
check("all 5 domains present",
      all(d in stats["domain_counts"] for d in DOMAINS))
check("domain counts sum to n",
      sum(stats["domain_counts"].values()) == 1000)
check("age_mean is float",             isinstance(stats["age_mean"], float))

# Distribution check — uniform random, so each degree ~250 out of 1000
for deg in DEGREES:
    c = stats["degree_counts"][deg]
    check(f"'{deg}' count roughly uniform ({c}/1000)",
          150 <= c <= 350, f"got {c}, expected ~250")

# ──────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("H. decode_vector() — label encoding inverse")
print("─"*60)

# Test every degree decodes correctly
for deg in DEGREES:
    orig = {**GOOD, "highest_degree": deg}
    vec  = encode_profile(orig)
    dec  = decode_vector(vec)
    check(f"degree '{deg}' round-trips",
          dec["highest_degree"] == deg,
          f"decoded as '{dec['highest_degree']}'")

# Test every domain decodes correctly
for dom in DOMAINS:
    orig = {**GOOD, "favourite_domain": dom}
    vec  = encode_profile(orig)
    dec  = decode_vector(vec)
    check(f"domain '{dom}' round-trips",
          dec["favourite_domain"] == dom,
          f"decoded as '{dec['favourite_domain']}'")

# Numeric round-trips
for age in [18, 25, 44, 69, 70]:
    orig = {**GOOD, "age": age}
    vec  = encode_profile(orig)
    dec  = decode_vector(vec)
    check(f"age={age} round-trips",
          abs(dec["age"] - age) <= 1, f"decoded as {dec['age']}")

# Wrong length raises ValueError
try:
    decode_vector([0.5] * 8)
    check("length=8 raises ValueError", False, "no exception")
except ValueError:
    check("length=8 raises ValueError", True)

try:
    decode_vector([0.5] * 12)
    check("length=12 raises ValueError", False,
          "no exception — old vector length must not be accepted")
except ValueError:
    check("length=12 raises ValueError (old format rejected)", True)

# ──────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print(f"  FINAL RESULT:  {PASS} PASSED   {FAIL} FAILED")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — ready for handoff")
else:
    print("  ❌ FIX ALL FAILURES BEFORE HANDING OFF TO TEAM")
print("═"*60)
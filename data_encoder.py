# =============================================================================
# FILE: data_encoder.py
# OWNER: PERSON 1 — Data Generation & Encoding
# PURPOSE: Generate synthetic learner profiles, encode them as 9-dimensional
#          numerical vectors, and provide utilities for saving/loading/decoding.
#
# This module is the foundation of the similarity search pipeline.
# Person 2 (baseline linear scan) and Person 3 (KD-tree) both import
# encode_profile() and encode_dataset() to convert raw profiles into
# numerical vectors before computing distances. Person 4 uses
# get_dataset_stats() and decode_vector() for the technical report.
#
# ENCODING DESIGN SUMMARY (9 dimensions):
#   [0]   age                  — min-max normalisation
#   [1]   income               — min-max normalisation
#   [2]   self_learning_hours  — min-max normalisation
#   [3]   highest_degree       — label encoding (ordinal)
#   [4-8] favourite_domain     — one-hot encoding (nominal)
#
# WHY TWO DIFFERENT ENCODING STRATEGIES?
#   We deliberately chose label encoding for degree and one-hot for domain
#   because the two attributes have fundamentally different natures:
#
#   - Degree has a natural ORDER: High School < Bachelor < Master < PhD.
#     Label encoding (0, 1/3, 2/3, 1) preserves this progression so that
#     the Euclidean distance between "High School" and "PhD" is larger
#     than between "Master" and "PhD" — matching real-world intuition.
#     If we used one-hot encoding instead, every pair of different degrees
#     would be equidistant (√2), losing the ordinal information entirely.
#
#   - Domain has NO natural order: "AI" is not higher or lower than
#     "Cybersecurity". One-hot encoding treats all pairs as equally
#     different, which is correct. If we used label encoding (e.g.,
#     AI=0, SoftEng=0.25, ...), we would falsely imply that AI is
#     "closer" to Software Engineering than to Business Analytics.
#
# STDLIB ONLY — no external packages (numpy, pandas, sklearn, etc.).
# =============================================================================

import random   # For reproducible synthetic data generation
import csv      # For reading/writing dataset CSV files
import os       # For checking file existence in get_or_create_dataset()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — These define the encoding contract for the entire project.
#             Persons 2, 3, and 4 depend on these values staying consistent.
#             Changing any of these requires updating ALL downstream modules.
# ─────────────────────────────────────────────────────────────────────────────

# Degree categories — ordered from lowest to highest qualification.
# The list order matters: the index position IS the label value used
# in encoding. For example, "High School" is at index 0 → label 0,
# "PhD" is at index 3 → label 3. After normalisation: 0/3=0.0, 3/3=1.0.
# DO NOT reorder this list — it would change every encoded vector.
DEGREES = ["High School", "Bachelor", "Master", "PhD"]

# Domain categories — no natural ordering exists between these fields.
# The list order defines which vector dimension (index 4–8) gets the 1.0
# in one-hot encoding. For example, "AI" is at index 0 → vector[4]=1.0,
# "Business Analytics" is at index 4 → vector[8]=1.0.
# DO NOT reorder this list — it would shift the one-hot positions.
DOMAINS = ["AI", "Software Engineering", "Data Science",
           "Cybersecurity", "Business Analytics"]

# Total dimensions in the encoded vector:
#   3 (numeric: age, income, hours)
# + 1 (degree: single label-encoded float)
# + 5 (domain: one-hot with 5 categories)
# = 9 dimensions total
# Note: The previous version used 12 dimensions (4 for degree one-hot).
# We reduced it to 9 by switching degree to label encoding.
VECTOR_LENGTH = 9

# Degree label lookup dictionary — maps degree string to its integer label.
# Using a dict gives O(1) lookup instead of DEGREES.index() which is O(n).
# With 100,000 profiles to encode, this small optimisation adds up.
# Result: {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
DEGREE_ORDER = {deg: i for i, deg in enumerate(DEGREES)}

# Maximum label value for degree normalisation.
# We divide by (len - 1) = 3, NOT by len = 4, so that PhD maps to
# exactly 1.0. Dividing by 4 would give PhD = 0.75, wasting the upper
# quarter of the [0, 1] range and compressing the distance between degrees.
DEGREE_MAX_LABEL = len(DEGREES) - 1  # = 3

# Normalisation bounds for numeric attributes.
# These define the min-max range for each attribute. Any value at the
# minimum maps to 0.0, any value at the maximum maps to 1.0.
# Used in both encode_profile() and decode_vector() for consistency.
AGE_MIN    = 18    # Minimum learner age in the dataset
AGE_MAX    = 70    # Maximum learner age in the dataset
INCOME_MIN = 5     # Minimum income (in thousands) in the dataset
INCOME_MAX = 100   # Maximum income (in thousands) in the dataset
HOURS_MIN  = 0.0   # Minimum daily self-learning hours
HOURS_MAX  = 4.0   # Maximum daily self-learning hours


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n=100_000, seed=42):
    """
    Generate n synthetic learner profiles with reproducible randomness.

    Each profile is a dict with 6 keys:
        id                  — unique integer identifier (0 to n-1)
        age                 — random int in [18, 70]
        income              — random int in [5, 100] (thousands)
        self_learning_hours — random float in [0.0, 4.0], rounded to 2 dp
        highest_degree      — random choice from DEGREES list
        favourite_domain    — random choice from DOMAINS list

    We use random.Random(seed) to create a LOCAL random number generator
    rather than calling random.randint() directly. This ensures that our
    output is fully deterministic regardless of what other code might do
    with the global random state. Two calls with the same n and seed will
    always produce identical datasets — essential for reproducible experiments.

    Args:
        n:    Number of profiles to generate (default 100,000 as required
              by the assignment for scalability testing).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of n profile dicts, with ids numbered 0 through n-1.
    """
    rng = random.Random(seed)  # Local RNG — isolated from global state
    dataset = []
    for i in range(n):
        profile = {
            "id":                   i,
            "age":                  rng.randint(AGE_MIN, AGE_MAX),
            "income":               rng.randint(INCOME_MIN, INCOME_MAX),
            # round to 2 decimal places to avoid floating-point noise
            "self_learning_hours":  round(rng.uniform(HOURS_MIN, HOURS_MAX), 2),
            "highest_degree":       rng.choice(DEGREES),
            "favourite_domain":     rng.choice(DOMAINS),
        }
        dataset.append(profile)
    print(f"[Person 1] Generated {n:,} profiles (seed={seed})")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# CSV PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(dataset, filepath="dataset.csv"):
    """
    Write the dataset to a CSV file with a header row.

    The CSV format is used because it is human-readable, easy to inspect,
    and does not require any external libraries to parse. The fieldnames
    are written in a fixed order so that the file structure is predictable
    for anyone who opens it.

    Args:
        dataset:  List of profile dicts (output of generate_dataset).
        filepath: Path to the output CSV file (default "dataset.csv").
    """
    fieldnames = ["id", "age", "income", "self_learning_hours",
                  "highest_degree", "favourite_domain"]
    # newline="" prevents csv.writer from inserting extra blank lines on Windows
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"[Person 1] Saved {len(dataset):,} profiles → {filepath}")


def load_dataset(filepath="dataset.csv"):
    """
    Load dataset from CSV, casting types back to their proper Python types.

    CSV stores every value as a string, so we must explicitly convert:
        id, age, income           → int   (whole numbers)
        self_learning_hours       → float (decimal number)
        highest_degree, favourite_domain → str (already correct as-is)

    Without these casts, downstream code (e.g., encode_profile) would
    receive strings like "25" instead of the integer 25, causing type
    errors in arithmetic operations.

    Args:
        filepath: Path to the CSV file to load (default "dataset.csv").

    Returns:
        List of profile dicts with correctly typed values.
    """
    dataset = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)  # Reads each row as an OrderedDict
        for row in reader:
            profile = {
                "id":                   int(row["id"]),
                "age":                  int(row["age"]),
                "income":               int(row["income"]),
                "self_learning_hours":  float(row["self_learning_hours"]),
                "highest_degree":       row["highest_degree"],
                "favourite_domain":     row["favourite_domain"],
            }
            dataset.append(profile)
    print(f"[Person 1] Loaded {len(dataset):,} profiles ← {filepath}")
    return dataset


def get_or_create_dataset(filepath="dataset.csv", n=100_000, seed=42):
    """
    Load the dataset from CSV if it already exists on disk.
    Otherwise, generate a fresh dataset, save it, and return it.

    This avoids re-generating 100,000 profiles every time the program
    runs, while still creating the file automatically on first run.

    Args:
        filepath: Path to the CSV file (default "dataset.csv").
        n:        Number of profiles to generate if file doesn't exist.
        seed:     Random seed for generation.

    Returns:
        List of profile dicts.
    """
    if os.path.exists(filepath):
        return load_dataset(filepath)
    dataset = generate_dataset(n=n, seed=seed)
    save_dataset(dataset, filepath)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_profile(profile):
    """
    Check that a profile dict has all required keys with valid values.

    This is called as the FIRST step in encode_profile() to catch bad
    data early with a clear error message, rather than letting it produce
    a silently wrong vector. We use ValueError (not assert) so that
    validation is never accidentally disabled by running Python with -O.

    Validation rules:
        - All 5 attribute keys must be present (id is not required)
        - age must be an integer in [18, 70]
        - income must be an integer in [5, 100]
        - self_learning_hours must be a float in [0.0, 4.0]
        - highest_degree must be one of the 4 values in DEGREES
        - favourite_domain must be one of the 5 values in DOMAINS

    Args:
        profile: Dict to validate.

    Returns:
        True if the profile passes all checks.

    Raises:
        ValueError: with a descriptive message on any violation.
    """
    # Check all required keys are present
    required = ["age", "income", "self_learning_hours",
                "highest_degree", "favourite_domain"]
    for key in required:
        if key not in profile:
            raise ValueError(f"Missing required key: '{key}'")

    # Check numeric ranges
    if not (AGE_MIN <= profile["age"] <= AGE_MAX):
        raise ValueError(
            f"age={profile['age']} out of range [{AGE_MIN}, {AGE_MAX}]")

    if not (INCOME_MIN <= profile["income"] <= INCOME_MAX):
        raise ValueError(
            f"income={profile['income']} out of range "
            f"[{INCOME_MIN}, {INCOME_MAX}]")

    if not (HOURS_MIN <= profile["self_learning_hours"] <= HOURS_MAX):
        raise ValueError(
            f"self_learning_hours={profile['self_learning_hours']} "
            f"out of range [{HOURS_MIN}, {HOURS_MAX}]")

    # Check categorical values are in the allowed lists (case-sensitive)
    if profile["highest_degree"] not in DEGREES:
        raise ValueError(
            f"highest_degree='{profile['highest_degree']}' not in {DEGREES}")

    if profile["favourite_domain"] not in DOMAINS:
        raise ValueError(
            f"favourite_domain='{profile['favourite_domain']}' not in {DOMAINS}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_profile(profile):
    """
    Convert a single profile dict into a 9-dimensional float vector.

    This is the core transformation that makes similarity search possible.
    Raw profiles contain a mix of integers, floats, and strings — which
    cannot be compared with Euclidean distance directly. This function
    maps every attribute into the [0.0, 1.0] range so that all dimensions
    contribute proportionally to the distance calculation.

    Vector layout (9 dimensions):
        Index  Attribute             Encoding method
        ─────  ────────────────────  ──────────────────────────────────
        [0]    age                   min-max: (age - 18) / 52
        [1]    income                min-max: (income - 5) / 95
        [2]    self_learning_hours   min-max: hours / 4.0
        [3]    highest_degree        label:   DEGREE_ORDER[deg] / 3
        [4]    domain = AI           one-hot: 1.0 if match, else 0.0
        [5]    domain = SoftEng      one-hot: 1.0 if match, else 0.0
        [6]    domain = DataSci      one-hot: 1.0 if match, else 0.0
        [7]    domain = Cyber        one-hot: 1.0 if match, else 0.0
        [8]    domain = BizAnal      one-hot: 1.0 if match, else 0.0

    WHY LABEL ENCODING FOR DEGREE (not one-hot):
        Education levels have a genuine ordinal progression:
        High School → Bachelor → Master → PhD. Each step represents
        additional years of study and a higher qualification. Label
        encoding preserves this ordering: the distance between
        "High School" (0.0) and "PhD" (1.0) is correctly three times
        larger than between "Master" (0.667) and "PhD" (1.0).
        One-hot encoding would treat ALL degree pairs as equidistant
        (each differing by √2), losing this meaningful relationship.

    WHY ONE-HOT ENCODING FOR DOMAIN (not label):
        Favourite domain has no natural ordering. There is no objective
        sense in which "AI" is higher or lower than "Cybersecurity".
        Assigning numeric labels (AI=0, SoftEng=1, ...) would create
        false distance relationships — implying AI is "closer" to
        Software Engineering than to Business Analytics. One-hot
        encoding correctly treats all domain pairs as equally different.

    WHY MIN-MAX NORMALISATION FOR NUMERIC ATTRIBUTES:
        Age ranges from 18 to 70, income from 5 to 100, hours from 0 to 4.
        Without normalisation, income (range 95) would dominate the distance
        calculation over hours (range 4) simply because of its larger scale.
        Min-max normalisation maps all three to [0, 1], giving each attribute
        equal weight in the distance metric.

    Args:
        profile: Dict with keys age, income, self_learning_hours,
                 highest_degree, favourite_domain.

    Returns:
        List of 9 floats, each in [0.0, 1.0].

    Raises:
        ValueError: if the profile fails validation.
    """
    # Validate first — catch bad data before it becomes a bad vector
    validate_profile(profile)

    vector = []

    # ── Index 0: age — min-max normalisation ──────────────────────────────
    # Formula: (value - min) / (max - min) → maps 18 to 0.0, 70 to 1.0
    vector.append((profile["age"] - AGE_MIN) / (AGE_MAX - AGE_MIN))

    # ── Index 1: income — min-max normalisation ───────────────────────────
    # Formula: (value - min) / (max - min) → maps 5 to 0.0, 100 to 1.0
    vector.append((profile["income"] - INCOME_MIN) / (INCOME_MAX - INCOME_MIN))

    # ── Index 2: self_learning_hours — min-max normalisation ──────────────
    # Since HOURS_MIN is 0.0, simplifies to: value / max → maps 0 to 0.0, 4 to 1.0
    vector.append(profile["self_learning_hours"] / HOURS_MAX)

    # ── Index 3: highest_degree — LABEL ENCODING ─────────────────────────
    # Step 1: Look up the integer label using the DEGREE_ORDER dict (O(1)).
    #         "High School"→0, "Bachelor"→1, "Master"→2, "PhD"→3
    # Step 2: Normalise by dividing by DEGREE_MAX_LABEL (3), so the values
    #         become: 0/3=0.0, 1/3≈0.333, 2/3≈0.667, 3/3=1.0
    # We use a dict lookup instead of DEGREES.index() for O(1) performance.
    # With 100,000 profiles to encode, this small optimisation adds up.
    label = DEGREE_ORDER[profile["highest_degree"]]
    vector.append(label / DEGREE_MAX_LABEL)

    # ── Indices 4–8: favourite_domain — ONE-HOT ENCODING ─────────────────
    # For each of the 5 domains, append 1.0 if it matches the profile's
    # domain, otherwise 0.0. This produces exactly one 1.0 and four 0.0s.
    # Example: "Cybersecurity" → [0.0, 0.0, 0.0, 1.0, 0.0]
    for domain in DOMAINS:
        vector.append(1.0 if profile["favourite_domain"] == domain else 0.0)

    # ── Sanity check ─────────────────────────────────────────────────────
    # Catches bugs early if someone modifies the encoding logic above
    assert len(vector) == VECTOR_LENGTH, \
        f"BUG: expected {VECTOR_LENGTH} dims, got {len(vector)}"

    return vector


def encode_dataset(dataset):
    """
    Encode all profiles in the dataset into vectors.

    WHY PRE-ENCODE?
    Encoding each profile on-the-fly during every distance call would
    re-encode each of the 100,000 profiles for every single query.
    For k-nearest-neighbour with multiple queries, that means encoding
    the same profile hundreds or thousands of times. Pre-encoding once
    at startup and caching the result eliminates this redundant work.

    Returns:
        List of (profile_id, vector) tuples — the format expected by
        baseline.py (Person 2) for linear scan and kdtree.py (Person 3)
        for tree construction. The tuple format keeps the ID attached
        to its vector so we can identify which profile a result belongs to.
    """
    encoded = [(p["id"], encode_profile(p)) for p in dataset]
    print(f"[Person 1] Encoded {len(encoded):,} profiles → "
          f"{VECTOR_LENGTH}-dim vectors")
    return encoded


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_stats(dataset):
    """
    Compute descriptive statistics over the raw dataset.

    Used by Person 4 to populate Section 2.1 of the technical report.
    Call this once after generating the dataset and include the output
    in the report's dataset statistics table.

    Returns:
        Dict with the following keys:
            n              — total number of profiles (int)
            age_min        — minimum age in the dataset (int)
            age_max        — maximum age in the dataset (int)
            age_mean       — mean age, rounded to 2 decimal places (float)
            income_min     — minimum income (int)
            income_max     — maximum income (int)
            income_mean    — mean income, rounded to 2 dp (float)
            hours_min      — minimum self-learning hours (float)
            hours_max      — maximum self-learning hours (float)
            hours_mean     — mean self-learning hours, rounded to 2 dp (float)
            degree_counts  — {degree_str: count} for ALL degrees in DEGREES
            domain_counts  — {domain_str: count} for ALL domains in DOMAINS

    Raises:
        ValueError: if the dataset is empty.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot compute stats on an empty dataset.")

    # Extract numeric columns into separate lists for min/max/mean
    ages    = [p["age"]                 for p in dataset]
    incomes = [p["income"]              for p in dataset]
    hours   = [p["self_learning_hours"] for p in dataset]

    # Initialise counts for ALL categories at zero first.
    # This guarantees every degree and domain appears in the output
    # even if its count happens to be 0 (e.g., in a small sample).
    degree_counts = {d: 0 for d in DEGREES}
    domain_counts = {d: 0 for d in DOMAINS}

    for p in dataset:
        degree_counts[p["highest_degree"]] += 1
        domain_counts[p["favourite_domain"]] += 1

    return {
        "n":           n,
        "age_min":     min(ages),
        "age_max":     max(ages),
        "age_mean":    round(sum(ages) / n, 2),
        "income_min":  min(incomes),
        "income_max":  max(incomes),
        "income_mean": round(sum(incomes) / n, 2),
        "hours_min":   min(hours),
        "hours_max":   max(hours),
        "hours_mean":  round(sum(hours) / n, 2),
        "degree_counts": degree_counts,
        "domain_counts": domain_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DECODING (inverse of encode_profile)
# ─────────────────────────────────────────────────────────────────────────────

def decode_vector(vector):
    """
    Reconstruct an approximate profile dict from a 9-dimensional encoded vector.

    This is the inverse of encode_profile(). Used by Person 4's experiments
    to display human-readable output alongside profile IDs and distances.

    Decoded values are clamped to valid ranges because floating-point
    arithmetic can produce values slightly outside expected bounds.

    Args:
        vector: List of exactly 9 floats.

    Returns:
        Profile dict with the 5 attribute fields (no 'id' key).

    Raises:
        ValueError: if vector length is not 9.
    """
    if len(vector) != VECTOR_LENGTH:
        raise ValueError(
            f"Expected vector of length {VECTOR_LENGTH}, got {len(vector)}."
        )

    # ── Index 0: age ─────────────────────────────────────────────────────
    age = round(vector[0] * (AGE_MAX - AGE_MIN)) + AGE_MIN
    age = min(AGE_MAX, max(AGE_MIN, age))

    # ── Index 1: income ──────────────────────────────────────────────────
    income = round(vector[1] * (INCOME_MAX - INCOME_MIN)) + INCOME_MIN
    income = min(INCOME_MAX, max(INCOME_MIN, income))

    # ── Index 2: self_learning_hours ─────────────────────────────────────
    hours = round(vector[2] * HOURS_MAX, 2)
    hours = min(HOURS_MAX, max(HOURS_MIN, hours))

    # ── Index 3: highest_degree (label decode) ───────────────────────────
    # Reverse of: label / DEGREE_MAX_LABEL
    label_idx = round(vector[3] * DEGREE_MAX_LABEL)
    label_idx = min(len(DEGREES) - 1, max(0, label_idx))
    degree = DEGREES[label_idx]

    # ── Indices 4–8: favourite_domain (argmax of one-hot) ────────────────
    domain_vals = vector[4:9]
    domain_idx  = domain_vals.index(max(domain_vals))
    domain = DOMAINS[domain_idx]

    return {
        "age":                 age,
        "income":              income,
        "self_learning_hours": hours,
        "highest_degree":      degree,
        "favourite_domain":    domain,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INLINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    """
    Quick built-in smoke tests. Run via: python data_encoder.py
    All tests must pass before handing off to the team.
    """
    print("\n[Person 1] Running built-in tests...")

    # Test 1: Dataset size
    ds = generate_dataset(n=500, seed=0)
    assert len(ds) == 500, "FAIL Test 1: wrong size"
    print("  ✅ Test 1 PASSED: Dataset size correct")

    # Test 2: Attribute value ranges
    for p in ds:
        assert 18 <= p["age"] <= 70
        assert 5 <= p["income"] <= 100
        assert 0.0 <= p["self_learning_hours"] <= 4.0
        assert p["highest_degree"] in DEGREES
        assert p["favourite_domain"] in DOMAINS
    print("  ✅ Test 2 PASSED: All attribute values in valid range")

    # Test 3: Vector length = 9
    for p in ds:
        vec = encode_profile(p)
        assert len(vec) == VECTOR_LENGTH, \
            f"FAIL Test 3: vector length {len(vec)}, expected {VECTOR_LENGTH}"
    print(f"  ✅ Test 3 PASSED: All vectors have length {VECTOR_LENGTH}")

    # Test 4: All values in [0.0, 1.0]
    for p in ds:
        for v in encode_profile(p):
            assert 0.0 <= v <= 1.0, f"FAIL Test 4: value {v} out of range"
    print("  ✅ Test 4 PASSED: All vector values in [0.0, 1.0]")

    # Test 5: Degree label encoding correct values
    degree_expected = {
        "High School": 0.0,
        "Bachelor":    round(1/3, 10),
        "Master":      round(2/3, 10),
        "PhD":         1.0,
    }
    for deg, expected in degree_expected.items():
        p   = {**ds[0], "highest_degree": deg}
        vec = encode_profile(p)
        assert abs(vec[3] - expected) < 1e-6, \
            f"FAIL Test 5: degree '{deg}' → {vec[3]}, expected {expected}"
    print("  ✅ Test 5 PASSED: Degree label encoding values correct")

    # Test 6: Domain one-hot sums to 1.0 (indices 4–8)
    for p in ds:
        vec = encode_profile(p)
        domain_sum = sum(vec[4:9])
        assert abs(domain_sum - 1.0) < 1e-9, \
            f"FAIL Test 6: domain one-hot sum = {domain_sum}"
    print("  ✅ Test 6 PASSED: Domain one-hot sums are all 1.0")

    # Test 7: Reproducibility
    ds_a = generate_dataset(n=100, seed=7)
    ds_b = generate_dataset(n=100, seed=7)
    assert ds_a == ds_b, "FAIL Test 7: same seed gave different datasets"
    print("  ✅ Test 7 PASSED: Same seed → identical dataset")

    print("\n[Person 1] All built-in tests PASSED ✅\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_tests()
    dataset = get_or_create_dataset("dataset.csv", n=100_000)
    stats   = get_dataset_stats(dataset)
    print(f"\nDataset stats:")
    print(f"  n={stats['n']:,}")
    print(f"  age    min={stats['age_min']}  max={stats['age_max']}  "
          f"mean={stats['age_mean']}")
    print(f"  income min={stats['income_min']}  max={stats['income_max']}  "
          f"mean={stats['income_mean']}")
    print(f"  hours  min={stats['hours_min']}  max={stats['hours_max']}  "
          f"mean={stats['hours_mean']}")
    print(f"\nDegree distribution:")
    for d, c in stats["degree_counts"].items():
        print(f"  {d:<15}: {c:,}  ({c/stats['n']*100:.1f}%)")
    print(f"\nDomain distribution:")
    for d, c in stats["domain_counts"].items():
        print(f"  {d:<25}: {c:,}  ({c/stats['n']*100:.1f}%)")

    sample = dataset[0]
    vec    = encode_profile(sample)
    dec    = decode_vector(vec)
    print(f"\nEncode/decode round-trip check on profile 0:")
    print(f"  Original: {sample}")
    print(f"  Vector:   {[round(v,4) for v in vec]}")
    print(f"  Decoded:  {dec}")
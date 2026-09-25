# Amazon_ML_Challenge
# Entity Resolution System

## 1. Project Overview

This project implements an **Entity Resolution (ER)** system for identifying records that refer to the **same real-world entity**.

The input data may contain multiple records representing the same business/entity, but the information may be inconsistent, incomplete, or formatted differently.

For example:

```text
Record 1:
ABC Technologies Pvt. Ltd.
12, M.G. Road, Hyderabad
+91 98765 43210

Record 2:
ABC TECHNOLOGIES PVT LTD
12 MG Rd Hyderabad
9876543210
```

Although these records are not exactly identical, they may represent the same entity.

Our system must determine whether two records refer to the same entity **using only the data provided by the problem**.

> **Important competition restriction:** We must NOT use external databases, APIs, geocoding services, business-registration databases, commercial entity-resolution APIs, or internet-based entity lookup to resolve identities. All matching decisions must be based on the provided dataset.

---

# 2. High-Level Architecture

The complete system will follow this pipeline:

```text
                         INPUT DATA
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Data Loading       │
                  │   & Profiling        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Preprocessing      │
                  │   & Normalization    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Candidate Generation │
                  │      / Blocking      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Field Similarity   │
                  │     Calculation      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Match Scoring      │
                  │  & Decision Engine   │
                  └──────────┬───────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
              CONFIDENT             UNCERTAIN /
                MATCH                NO MATCH
                  │
                  ▼
          ┌─────────────────┐
          │ Entity ID       │
          │ Assignment      │
          └────────┬────────┘
                   │
                   ▼
              FINAL OUTPUT
```

---

# 3. Why Entity Resolution?

Traditional database matching often assumes that two records are equal when their fields are equal.

For example:

```text
name = "ABC Technologies"
```

and

```text
name = "ABC Technologies"
```

can be matched using exact equality.

But real-world data is usually messy:

```text
ABC Technologies Pvt Ltd
ABC TECHNOLOGIES PRIVATE LIMITED
A.B.C. Technologies
ABC Tech Pvt. Ltd.
```

Similarly, addresses, phone numbers, emails, and other attributes can have formatting differences.

Therefore, we need to determine **similarity rather than simple equality**.

---

# 4. Complete Data Flow

The system processes every record through several stages.

```text
Raw Record
    │
    ▼
Normalization
    │
    ▼
Normalized Record
    │
    ▼
Candidate Generation
    │
    ▼
Potential Matching Records
    │
    ▼
Field-Level Similarity
    │
    ▼
Combined Match Score
    │
    ▼
Decision
    │
    ├───────────────┐
    ▼               ▼
 MATCH          NO MATCH
    │
    ▼
Entity ID
```

---

# 5. Stage 1 — Data Loading & Profiling

## Purpose

Before implementing the matching algorithm, we need to understand the supplied dataset.

We need to determine:

* What files are provided?
* How many records exist?
* What columns are available?
* Which fields contain identifying information?
* Which fields contain missing values?
* Which fields are duplicated?
* What is the expected output format?
* Are labels/ground truth provided?
* How large is the dataset?

Example:

```text
Input Record

record_id
name
address
phone
email
website
...
```

The actual fields will be determined from the supplied dataset.

---

# 6. Stage 2 — Data Preprocessing & Normalization

## Purpose

Records cannot be compared effectively if the same information is represented in different formats.

Normalization converts different representations into a more consistent form.

Example:

```text
ABC Technologies Pvt. Ltd.
```

and

```text
ABC TECHNOLOGIES PVT LTD
```

can be transformed into comparable representations.

---

## 6.1 Name Normalization

Possible operations:

```text
Original
   ↓
Lowercase
   ↓
Unicode normalization
   ↓
Remove unnecessary punctuation
   ↓
Normalize whitespace
   ↓
Handle known business suffixes
```

Example:

```text
"ABC Technologies Pvt. Ltd."
              ↓
"abc technologies"
```

We must avoid overly aggressive normalization that removes meaningful information.

---

## 6.2 Phone Normalization

Possible transformations:

```text
+91 98765 43210
+91-98765-43210
9876543210
```

can potentially be converted into a consistent representation.

The exact rules will depend on the supplied data.

---

## 6.3 Email Normalization

Basic normalization may include:

```text
ABC@Example.com
```

↓

```text
abc@example.com
```

We should avoid modifying the email structure in ways that could create false matches.

---

## 6.4 Address Normalization

Addresses can vary significantly.

Example:

```text
12, M.G. Road, Hyderabad
```

vs.

```text
12 MG Rd Hyderabad
```

Possible operations:

* lowercase
* punctuation normalization
* whitespace normalization
* common abbreviation handling
* token normalization

We must NOT use external geocoding APIs.

---

# 7. Stage 3 — Candidate Generation / Blocking

## Why is candidate generation necessary?

Suppose we have:

```text
1,000,000 records
```

Comparing every record against every other record would require an enormous number of comparisons.

A naive approach would look like:

```text
Record A
   │
   ├── Record B
   ├── Record C
   ├── Record D
   ├── ...
   └── Record 1,000,000
```

This is inefficient.

Instead, we first find a smaller set of **potential candidates**.

```text
Record A
   │
   ▼
Blocking
   │
   ├── Candidate 1
   ├── Candidate 2
   ├── Candidate 3
   └── Candidate 4
```

Then detailed similarity calculations are performed only on those candidates.

---

# 8. Blocking Strategy

Blocking creates groups of records that are likely to refer to the same entity.

Potential blocking keys include:

```text
Normalized name tokens
Phone number
Phone suffix
Email domain
Address tokens
Postal/ZIP code
Other fields available in the dataset
```

We can use multiple blocking strategies.

Example:

```text
Block 1:
same normalized phone

OR

Block 2:
same important name token

OR

Block 3:
same address/postal information
```

The goal is:

```text
Reduce comparisons
        +
Maintain high candidate recall
```

We should avoid overly restrictive blocking because a correct match might never reach the matching stage.

---

# 9. Stage 4 — Field-Level Similarity

After candidate generation, each input record is compared with each candidate.

For example:

```text
Record A                       Record B

name       ────────────────── name
address    ────────────────── address
phone      ────────────────── phone
email      ────────────────── email
```

Each field receives its own similarity score.

Example:

```text
Name similarity       = 0.94
Address similarity    = 0.82
Phone similarity      = 1.00
Email similarity      = 1.00
```

The actual similarity algorithms will be selected during experimentation.

---

# 10. Similarity Algorithms

Possible techniques include:

### Exact Matching

Useful for highly reliable fields.

```text
A == B
```

Result:

```text
1.0 → exact match
0.0 → different
```

---

### Edit Distance

Measures how many edits are required to transform one string into another.

Useful for:

```text
"technologies"
"technology"
```

---

### Jaro-Winkler

Useful for comparing similar strings, particularly names.

Example:

```text
"abc technologies"
"abc technology"
```

---

### Jaccard Similarity

Useful for comparing sets/tokens.

Example:

```text
A = {abc, technologies, hyderabad}

B = {abc, technologies, hyd}
```

---

### TF-IDF / Cosine Similarity

Can be considered for larger text fields such as addresses or names.

We will evaluate which methods actually improve performance on the provided dataset.

---

# 11. Stage 5 — Match Scoring

Individual field similarities need to be combined into an overall score.

Conceptually:

```text
                Name similarity
                       │
                       ▼
                Address similarity
                       │
                       ▼
                 Phone similarity
                       │
                       ▼
                 Email similarity
                       │
                       ▼
              ┌─────────────────┐
              │ Weighted Score  │
              └────────┬────────┘
                       │
                       ▼
                 Final Score
```

A possible initial model is:

```text
score =
    w1 × name_similarity
  + w2 × address_similarity
  + w3 × phone_similarity
  + w4 × email_similarity
```

Where:

```text
w1 + w2 + w3 + w4 = 1
```

The weights are NOT fixed yet.

They will be determined through experimentation and evaluation.

---

# 12. Missing Values

Missing data must be handled carefully.

Example:

```text
Record A

name     = ABC Technologies
phone    = 9876543210
email    = NULL
address  = Hyderabad
```

If Record B has an email, we should not automatically consider the missing email as a strong negative signal.

Instead:

```text
Available fields
       ↓
Calculate similarities
       ↓
Use available evidence
       ↓
Calculate normalized score
```

The implementation must distinguish between:

```text
Fields are different
```

and

```text
Field is missing
```

These are not the same thing.

---

# 13. Stage 6 — Decision Engine

The final score is converted into a decision.

Conceptually:

```text
                    Final Score
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
          HIGH        MEDIUM        LOW
             │           │           │
             ▼           ▼           ▼
           MATCH      UNCERTAIN    NO MATCH
```

Example thresholds:

```text
score >= T_high
        → MATCH

T_low <= score < T_high
        → UNCERTAIN

score < T_low
        → NO MATCH
```

These thresholds are examples only.

They must be selected using evaluation results.

---

# 14. Entity Assignment

If a record is determined to match an existing entity:

```text
Incoming Record
       │
       ▼
Best Candidate
       │
       ▼
Existing Entity ID
```

Example:

```text
Input:

ABC Technologies Pvt Ltd

        ↓

Matched Entity:

entity_id = 1042
```

The final output should contain the required entity identifier and other fields specified by the problem statement.

---

# 15. Handling Ambiguous Matches

Sometimes multiple candidates may have similar scores.

Example:

```text
Candidate A → 0.91
Candidate B → 0.90
Candidate C → 0.42
```

We should not blindly assume Candidate A is correct.

The system should consider:

```text
Best score
+
Difference between best and second-best
+
Field-level evidence
+
Missing information
```

For example:

```text
Best score = 0.91
Second best = 0.90
```

is much more ambiguous than:

```text
Best score = 0.95
Second best = 0.55
```

This can be incorporated into the decision logic.

---

# 16. Evaluation

The evaluation component determines whether our system is actually improving.

Depending on the available ground truth and competition evaluation, we will measure appropriate metrics such as:

```text
Precision
Recall
F1-score
Accuracy
False Positives
False Negatives
```

---

## False Positive

Two different entities are incorrectly merged.

```text
Entity A
   │
   ▼
incorrectly matched
   │
   ▼
Entity B
```

This is a dangerous error because unrelated entities can become one entity.

---

## False Negative

Two records belonging to the same entity are not matched.

```text
Record A ─────X───── Record B
       same entity
```

The system failed to identify the relationship.

---

# 17. Experimentation Strategy

We will build a simple baseline first.

### Experiment 1

```text
Exact matching
```

### Experiment 2

```text
Basic normalization
+
Exact matching
```

### Experiment 3

```text
Normalization
+
Candidate generation
+
String similarity
```

### Experiment 4

```text
Normalization
+
Blocking
+
Multiple field similarities
+
Weighted score
```

### Experiment 5

```text
Improved blocking
+
Improved similarity
+
Optimized weights
+
Optimized thresholds
```

We should keep experiment results so that we can understand which changes actually improve the system.

---

# 18. Project Architecture

Recommended project structure:

```text
entity-resolution/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── preprocessing/
│   ├── __init__.py
│   ├── name.py
│   ├── address.py
│   ├── phone.py
│   ├── email.py
│   └── normalize.py
│
├── matching/
│   ├── __init__.py
│   ├── blocking.py
│   ├── candidate_generation.py
│   ├── similarity.py
│   └── scoring.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── experiments.py
│   └── threshold_analysis.py
│
├── pipeline/
│   ├── __init__.py
│   └── resolver.py
│
├── tests/
│   ├── test_normalization.py
│   ├── test_blocking.py
│   ├── test_similarity.py
│   └── test_resolver.py
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# 19. Team Responsibilities

## Teammate 1 — Architecture & Integration

Responsibilities:

* Understand complete problem
* Define overall architecture
* Create integration pipeline
* Integrate all modules
* Maintain README/documentation
* Review pull requests
* Run final end-to-end testing

Main files:

```text
pipeline/
resolver.py
```

---

## Teammate 2 — Preprocessing & Normalization

Responsibilities:

* Analyze raw data
* Handle missing values
* Normalize names
* Normalize addresses
* Normalize phone numbers
* Normalize emails
* Write unit tests

Main files:

```text
preprocessing/
```

---

## Teammate 3 — Candidate Generation & Matching

Responsibilities:

* Implement blocking
* Candidate generation
* Similarity functions
* Field-level comparison
* Match scoring
* Ambiguous match handling

Main files:

```text
matching/
```

---

## Teammate 4 — Evaluation & Optimization

Responsibilities:

* Build evaluation framework
* Calculate metrics
* Run experiments
* Tune thresholds
* Compare similarity methods
* Analyze false positives/false negatives
* Document experiment results

Main files:

```text
evaluation/
```

---

# 20. Git Workflow

Each teammate should work on a separate branch.

```text
main
 │
 ├── feature/preprocessing
 │
 ├── feature/matching
 │
 ├── feature/evaluation
 │
 └── feature/integration
```

Recommended workflow:

```bash
git checkout -b feature/preprocessing
```

Work:

```bash
git add .
git commit -m "Add name and phone normalization"
git push origin feature/preprocessing
```

Then create a Pull Request.

Do not directly push experimental changes into `main`.

---

# 21. Integration Contract

To avoid conflicts between teammates, modules should communicate using clearly defined interfaces.

For example:

```python
normalize_record(record)
```

returns:

```python
{
    "name": "...",
    "address": "...",
    "phone": "...",
    "email": "..."
}
```

Candidate generation:

```python
generate_candidates(record, reference_data)
```

returns:

```python
[
    candidate_1,
    candidate_2,
    candidate_3
]
```

Similarity:

```python
calculate_similarity(record_a, record_b)
```

returns:

```python
{
    "name": 0.92,
    "address": 0.81,
    "phone": 1.0,
    "email": 1.0
}
```

Scoring:

```python
calculate_match_score(similarities)
```

returns:

```text
0.93
```

Decision:

```python
make_decision(score)
```

returns:

```text
MATCH
```

This makes integration much easier.

---

# 22. Development Order

We should implement the system in this order:

```text
STEP 1
Understand dataset
        ↓
STEP 2
Profile data
        ↓
STEP 3
Implement normalization
        ↓
STEP 4
Create baseline exact matching
        ↓
STEP 5
Implement blocking
        ↓
STEP 6
Implement candidate generation
        ↓
STEP 7
Implement similarity functions
        ↓
STEP 8
Implement scoring
        ↓
STEP 9
Implement decision thresholds
        ↓
STEP 10
Evaluate
        ↓
STEP 11
Optimize
        ↓
STEP 12
End-to-end testing
        ↓
STEP 13
Final submission
```

---

# 23. Important Principles

### 1. No external entity lookup

Do not use:

```text
Google Search
Google Maps
Geocoding APIs
Government databases
Business registries
Commercial entity-resolution APIs
External company databases
```

The system must operate on the provided data.

---

### 2. Don't start with a complex ML model

First establish a strong baseline:

```text
Normalization
+
Blocking
+
String similarity
+
Weighted scoring
```

Only introduce more complex approaches if experiments show that they improve the result.

---

### 3. Every change should be measurable

Do not say:

```text
"This algorithm looks better."
```

Instead measure:

```text
Before:
F1 = X

After:
F1 = Y
```

and understand why it changed.

---

### 4. Preserve raw data

Never overwrite the original dataset.

Use:

```text
data/raw/
```

for original files and:

```text
data/processed/
```

for transformed data.

---

### 5. Keep components independent

The normalization module should not contain matching logic.

The matching module should not contain evaluation logic.

The evaluation module should not modify the core resolver.

This makes experimentation easier.

---

# 24. Final Architecture

The final system should look like:

```text
                         ┌─────────────────┐
                         │   RAW DATA      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ DATA PROFILING  │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ PREPROCESSING /          │
                    │ NORMALIZATION            │
                    │                          │
                    │ Name                     │
                    │ Address                  │
                    │ Phone                    │
                    │ Email                    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ BLOCKING / CANDIDATE      │
                    │ GENERATION                │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ FIELD SIMILARITY         │
                    │                          │
                    │ Name                     │
                    │ Address                  │
                    │ Phone                    │
                    │ Email                    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ MATCH SCORE              │
                    │                          │
                    │ Weighted / Learned       │
                    │ Combination              │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ DECISION ENGINE          │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
                 MATCH       UNCERTAIN    NO MATCH
                    │
                    ▼
             ┌───────────────┐
             │ ENTITY ID     │
             │ ASSIGNMENT    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ FINAL OUTPUT  │
             └───────────────┘

                     ▲
                     │
              ┌──────┴──────┐
              │ EVALUATION  │
              │             │
              │ Precision   │
              │ Recall      │
              │ F1          │
              │ Errors      │
              └─────────────┘
```

---

# 25. First Team Meeting Checklist

Before coding, all four teammates must agree on:

* [ ] Dataset files identified
* [ ] Input columns identified
* [ ] Output format identified
* [ ] Ground truth/labels identified
* [ ] Missing-value patterns understood
* [ ] Restrictions understood
* [ ] Normalization strategy agreed
* [ ] Blocking strategy discussed
* [ ] Similarity methods selected for baseline
* [ ] Evaluation metrics identified
* [ ] Git repository created
* [ ] Branches created
* [ ] Responsibilities assigned

## First milestone

The first milestone is **NOT a highly accurate matcher**.

The first milestone is:

```text
Raw Dataset
     ↓
Normalized Dataset
     ↓
Candidate Pairs
     ↓
Similarity Scores
     ↓
Match/No-Match
     ↓
Evaluation
```

Once this basic pipeline works end-to-end, we improve each stage independently.

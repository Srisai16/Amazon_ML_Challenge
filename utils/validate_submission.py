#!/usr/bin/env python3
"""
Submission Validator for Amazon ML Challenge 2026: Business Entity Resolution
Standard Library Only (no external dependencies required).

Usage:
  python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
"""

import argparse
import os
import sys


def read_tsv_file(filepath):
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"
    
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if len(header) < 2:
            return None, f"Header line must have at least 2 columns separated by tab, got: {header}"
        
        col1, col2 = header[0], header[1]
        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                entity_id = parts[0]
                matched_ids = []
            elif len(parts) >= 2:
                entity_id = parts[0]
                matched_raw = parts[1].strip()
                matched_ids = [m.strip() for m in matched_raw.split(",") if m.strip()] if matched_raw else []
            else:
                continue

            if entity_id in data:
                return None, f"Duplicate entity_id '{entity_id}' found at line {line_num}"
            data[entity_id] = matched_ids
            
    return (col1, col2, data), None


def load_test_ids(test_dir):
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")

    for fpath in [s1_file, s2_file, s3_file]:
        if not os.path.exists(fpath):
            return None, None, None, f"Test file not found: {fpath}"

    s1_ids = set()
    s2_ids = set()
    s3_ids = set()

    def get_ids_from_tsv(path):
        ids = set()
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts and len(parts) > id_idx:
                    ids.add(parts[id_idx].strip())
        return ids

    try:
        s1_ids = get_ids_from_tsv(s1_file)
        s2_ids = get_ids_from_tsv(s2_file)
        s3_ids = get_ids_from_tsv(s3_file)
    except Exception as e:
        return None, None, None, f"Error reading test IDs: {str(e)}"

    return s1_ids, s2_ids, s3_ids, None


def validate(matching_path, candidate_path, test_dir):
    errors = []
    warnings = []

    print("=" * 70)
    print("🔍 Amazon ML Challenge 2026: Submission Validation")
    print("=" * 70)

    # 1. Read Test IDs
    s1_ids, s2_ids, s3_ids, err = load_test_ids(test_dir)
    if err:
        errors.append(f"Test Directory Error: {err}")
        return errors, warnings

    valid_match_targets = s2_ids.union(s3_ids)
    print(f"📊 Test Reference Entities (S1): {len(s1_ids)}")
    print(f"📊 Test Target Entities (S2 + S3): {len(valid_match_targets)} (S2: {len(s2_ids)}, S3: {len(s3_ids)})")

    # 2. Validate matching_results.tsv
    print(f"\nChecking matching_results.tsv: {matching_path}")
    res_m, err_m = read_tsv_file(matching_path)
    if err_m:
        errors.append(f"matching_results.tsv Error: {err_m}")
    else:
        col1, col2, m_data = res_m
        if col1 != "source1_entity_id" or col2 != "matched_entity_ids":
            errors.append(f"matching_results.tsv header must be 'source1_entity_id\\tmatched_entity_ids', got '{col1}\\t{col2}'")

        # Check coverage of all S1 test IDs
        m_s1_keys = set(m_data.keys())
        missing_s1 = s1_ids - m_s1_keys
        extra_s1 = m_s1_keys - s1_ids

        if missing_s1:
            errors.append(f"matching_results.tsv is missing {len(missing_s1)} S1 entities (e.g., {list(missing_s1)[:3]})")
        if extra_s1:
            errors.append(f"matching_results.tsv contains {len(extra_s1)} unknown S1 entities not in test_source1 (e.g., {list(extra_s1)[:3]})")

        # Check matched ID integrity
        for s1, targets in m_data.items():
            if len(targets) != len(set(targets)):
                errors.append(f"matching_results.tsv: Duplicate target IDs found for entity '{s1}'")
            for t in targets:
                if t.startswith("S1-"):
                    errors.append(f"matching_results.tsv: Self-match / S1 target '{t}' found for entity '{s1}'")
                elif t not in valid_match_targets:
                    errors.append(f"matching_results.tsv: Target '{t}' for entity '{s1}' does not exist in S2 or S3 test sets")

    # 3. Validate candidate_pairs.tsv
    print(f"\nChecking candidate_pairs.tsv: {candidate_path}")
    res_c, err_c = read_tsv_file(candidate_path)
    if err_c:
        errors.append(f"candidate_pairs.tsv Error: {err_c}")
    else:
        col1, col2, c_data = res_c
        if col1 != "source1_entity_id" or col2 != "candidate_entity_ids":
            errors.append(f"candidate_pairs.tsv header must be 'source1_entity_id\\tcandidate_entity_ids', got '{col1}\\t{col2}'")

        c_s1_keys = set(c_data.keys())
        missing_c_s1 = s1_ids - c_s1_keys
        if missing_c_s1:
            errors.append(f"candidate_pairs.tsv is missing {len(missing_c_s1)} S1 entities (e.g., {list(missing_c_s1)[:3]})")

        for s1, cands in c_data.items():
            if len(cands) != len(set(cands)):
                errors.append(f"candidate_pairs.tsv: Duplicate candidate IDs found for entity '{s1}'")
            for c in cands:
                if c.startswith("S1-"):
                    errors.append(f"candidate_pairs.tsv: S1 candidate '{c}' found for entity '{s1}'")
                elif c not in valid_match_targets:
                    errors.append(f"candidate_pairs.tsv: Candidate '{c}' for entity '{s1}' does not exist in S2/S3 test sets")

        # 4. Cross-check matching results is a subset of candidates
        if not err_m:
            for s1, matches in m_data.items():
                cands_set = set(c_data.get(s1, []))
                for m in matches:
                    if m not in cands_set:
                        warnings.append(f"Pipeline Consistency Warning: Match '{m}' for '{s1}' was not present in candidate_pairs.tsv")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Submission Validator")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()

    errors, warnings = validate(args.matching, args.candidate, args.test_dir)

    print("\n" + "=" * 70)
    if warnings:
        print(f"⚠️  {len(warnings)} WARNING(S) FOUND:")
        for w in warnings[:10]:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings.")

    if errors:
        print(f"\n❌ VALIDATION FAILED! {len(errors)} ERROR(S):")
        for e in errors[:15]:
            print(f"  - {e}")
        if len(errors) > 15:
            print(f"  ... and {len(errors) - 15} more errors.")
        sys.exit(1)
    else:
        print("✅ PASS: All format and integrity checks passed successfully (exit 0)!")
        print("Ready for upload to portal & packaging into submission zip.")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
FAST Threshold Re-Optimizer + Inference Script

ROOT CAUSE ANALYSIS:
- Submission #2 scored 0.851418 (barely improved from 0.848696)
- We predicted 10.0% singletons but ground truth is only ~5.6%
- We predicted 2.90 matches/entity but ground truth is ~3.67

FIXES:
1. Re-run threshold grid search with wider/aggressive grid
2. Immediately run full test inference with the improved policy
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.blocking.blocker import RareTokenBlocker
from src.data.loader import fld
from src.evaluation.metrics import calculate_entity_f_beta
from src.features.feature_extractor import FeatureExtractor
from src.models.gbdt_ranker import GBDTRanker

CACHE_ROOT = r'S:\Amazon_ML_Challenge\cache'
MODEL_PATH = r'S:\Amazon_ML_Challenge\saved_models\gbdt_model.pkl'
POLICY_PATH = r'S:\Amazon_ML_Challenge\saved_models\threshold_policy.json'
OUTPUT_DIR = r'S:\Amazon_ML_Challenge\output'


def optimize_thresholds_aggressive(q_val, t_val, val_probs, val_gt, val_queries):
    print('\n=== AGGRESSIVE THRESHOLD GRID SEARCH ===')
    t0 = time.time()

    q_to_pairs = {q: [] for q in val_queries}
    for q, t, p in zip(q_val.tolist(), t_val.tolist(), val_probs.tolist()):
        if q in q_to_pairs:
            q_to_pairs[q].append((t, p))

    total_queries = len(val_queries)
    true_singletons = sum(1 for q in val_queries if not val_gt.get(q))
    print(f'True singletons in val: {true_singletons:,} ({true_singletons/total_queries:.2%})')

    max_probs_non_sing = [max((p for _,p in q_to_pairs.get(q,[])), default=0.0)
                          for q in val_queries if val_gt.get(q)]
    if max_probs_non_sing:
        arr = np.array(max_probs_non_sing)
        for pct in [5,10,20]:
            print(f'  p{pct} max-prob for true non-singletons: {np.percentile(arr,pct):.4f}')
        for t in [0.05,0.10,0.15,0.20,0.30,0.40]:
            print(f'  Fraction true non-singletons with max_p<{t}: {(arr<t).mean():.2%}')

    best_f05 = -1.0
    best_t_sing = 0.40
    best_t_match = 0.625

    t_sing_grid = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    t_match_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.52, 0.55, 0.57, 0.60, 0.625, 0.65, 0.68, 0.70, 0.75]

    print(f'Running {len(t_sing_grid) * len(t_match_grid)} grid combos...')
    for t_sing in t_sing_grid:
        best_f_for_sing = -1.0
        best_tm = t_match_grid[0]
        for t_match in t_match_grid:
            scores = []
            for q in val_queries:
                true_s = val_gt.get(q, set())
                pairs = q_to_pairs.get(q, [])
                if not pairs:
                    pred_s = set()
                else:
                    max_p = max(p for _, p in pairs)
                    pred_s = set() if max_p < t_sing else {tgt for tgt, p in pairs if p >= t_match}
                scores.append(calculate_entity_f_beta(true_s, pred_s, beta=0.5))
            f05 = float(np.mean(scores))
            if f05 > best_f_for_sing:
                best_f_for_sing = f05
                best_tm = t_match
            if f05 > best_f05:
                best_f05 = f05
                best_t_sing = float(t_sing)
                best_t_match = float(t_match)
        print(f'  T_sing={t_sing:.2f} -> best T_match={best_tm:.3f} F0.5={best_f_for_sing:.5f}')

    print(f'Grid done in {time.time()-t0:.1f}s')
    print(f'OPTIMAL: T_sing={best_t_sing:.3f} T_match={best_t_match:.3f} F0.5={best_f05:.5f}')

    pred_sing_opt = sum(1 for q in val_queries
                        if not q_to_pairs.get(q) or max(p for _,p in q_to_pairs[q]) < best_t_sing)
    print(f'Predicted singletons at optimal: {pred_sing_opt:,} ({pred_sing_opt/total_queries:.2%}) vs true {true_singletons:,} ({true_singletons/total_queries:.2%})')
    return best_t_sing, best_t_match, best_f05


def run_full_inference(t_singleton, t_match, top_k=35):
    print(f'\n=== FULL TEST INFERENCE T_sing={t_singleton:.3f} T_match={t_match:.3f} ===')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matching_out = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    candidate_out = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

    ranker = GBDTRanker()
    ranker.load(MODEL_PATH)
    extractor = FeatureExtractor()

    s1_dir = os.path.join(CACHE_ROOT, 'test', 's1')
    s2_dir = os.path.join(CACHE_ROOT, 'test', 's2')
    s3_dir = os.path.join(CACHE_ROOT, 'test', 's3')
    s1_ids = np.load(fld(s1_dir, 'id'), mmap_mode='r')
    s2_ids = np.load(fld(s2_dir, 'id'), mmap_mode='r')
    s3_ids = np.load(fld(s3_dir, 'id'), mmap_mode='r')
    n_s1 = len(s1_ids)
    n2 = len(s2_ids)
    n_targets = n2 + len(s3_ids)

    blocker = RareTokenBlocker(CACHE_ROOT, 'test', top_k=top_k).build(force=False, verbose=False)
    print(f'Test: S1={n_s1:,} | Targets={n_targets:,}')

    def decode_tgt(idx):
        return (s2_ids[idx] if idx < n2 else s3_ids[idx-n2]).decode('utf-8','ignore').strip()

    t0 = time.time()
    total_cands = total_matches = total_singletons = 0
    block_size = 10_000

    with open(candidate_out, 'w', encoding='utf-8') as fc, open(matching_out, 'w', encoding='utf-8') as fm:
        fc.write('source1_entity_id\tcandidate_entity_ids\n')
        fm.write('source1_entity_id\tmatched_entity_ids\n')
        for lo in range(0, n_s1, block_size):
            hi = min(lo + block_size, n_s1)
            b_len = hi - lo
            r, c, v = blocker.block_range(lo, hi)
            cands_pq = {i: [] for i in range(b_len)}
            for lq, ti in zip(r.tolist(), c.tolist()):
                cands_pq[lq].append(ti)
            matched_pq = {i: [] for i in range(b_len)}
            if len(r) > 0:
                qg = r + lo
                X = extractor.extract_batch_from_cache(CACHE_ROOT, 'test', qg, c, v)
                probs = ranker.predict_proba(X)
                qcp = {i: [] for i in range(b_len)}
                for lq, ti, p in zip(r.tolist(), c.tolist(), probs.tolist()):
                    qcp[lq].append((ti, p))
                for lq, pairs in qcp.items():
                    if not pairs: continue
                    max_p = max(p for _, p in pairs)
                    if max_p >= t_singleton:
                        matched_pq[lq] = [t for t, p in pairs if p >= t_match]
            for li in range(b_len):
                gq = lo + li
                s1e = s1_ids[gq].decode('utf-8','ignore').strip()
                cu = list(dict.fromkeys(decode_tgt(t) for t in cands_pq[li]))
                mu = list(dict.fromkeys(decode_tgt(t) for t in matched_pq[li]))
                total_cands += len(cu)
                total_matches += len(mu)
                if not mu: total_singletons += 1
                fc.write(f'{s1e}\t{",".join(cu)}\n')
                fm.write(f'{s1e}\t{",".join(mu)}\n')
            if (hi % 200_000 == 0) or (hi == n_s1):
                el = time.time() - t0
                eta = (n_s1-hi)/max(1, hi/max(1e-5,el))
                print(f'  {hi:,}/{n_s1:,} ({hi/n_s1:.0%}) {hi/max(1e-5,el):,.0f}q/s Cands/q={total_cands/hi:.1f} Matches/q={total_matches/hi:.2f} Singletons={total_singletons/hi:.2%} ETA={eta/60:.1f}m', flush=True)

    el = time.time() - t0
    print(f'Inference done in {el:.0f}s')
    print(f'Singletons: {total_singletons:,} ({total_singletons/n_s1:.2%}) [target ~5.6%]')
    print(f'Avg matches/non-singleton: {total_matches/max(1,n_s1-total_singletons):.2f} [target ~3.67]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-val-queries', type=int, default=300_000)
    parser.add_argument('--top-k', type=int, default=35)
    parser.add_argument('--skip-retune', action='store_true')
    parser.add_argument('--t-singleton', type=float, default=0.40)
    parser.add_argument('--t-match', type=float, default=0.625)
    args = parser.parse_args()

    if args.skip_retune:
        t_sing, t_match = args.t_singleton, args.t_match
    else:
        SPLIT = 'train'
        s1_path = os.path.join(CACHE_ROOT, SPLIT, 's1')
        s2_path = os.path.join(CACHE_ROOT, SPLIT, 's2')
        s3_path = os.path.join(CACHE_ROOT, SPLIT, 's3')
        gt_path = os.path.join(CACHE_ROOT, SPLIT, 'gt')

        n_s1_train = len(np.load(fld(s1_path, 'idh'), mmap_mode='r'))
        n2 = len(np.load(fld(s2_path, 'idh'), mmap_mode='r'))
        n3 = len(np.load(fld(s3_path, 'idh'), mmap_mode='r'))
        n_targets = n2 + n3
        pos_keys = np.load(fld(gt_path, 'pos_keys'))

        n_val = min(args.n_val_queries, n_s1_train)
        start_q = max(150_000, n_s1_train - n_val)
        val_queries = set(range(start_q, start_q + n_val))
        print(f'Val queries: {len(val_queries):,} (S1 idx {start_q} to {start_q+n_val})')

        ranker = GBDTRanker()
        ranker.load(MODEL_PATH)
        extractor = FeatureExtractor()

        blocker = RareTokenBlocker(CACHE_ROOT, SPLIT, top_k=args.top_k).build(force=False, verbose=False)

        print('Generating blocking candidates for validation queries...')
        t0 = time.time()
        q_rows_l, t_rows_l, scores_l = [], [], []
        step = 5000
        for lo in range(start_q, start_q + n_val, step):
            hi = min(lo + step, start_q + n_val)
            r, c, v = blocker.block_range(lo, hi)
            if len(r) > 0:
                q_rows_l.append(r + lo)
                t_rows_l.append(c)
                scores_l.append(v)
        all_q = np.concatenate(q_rows_l)
        all_t = np.concatenate(t_rows_l)
        all_scores = np.concatenate(scores_l)
        print(f'Candidates: {len(all_q):,} ({len(all_q)/n_val:.1f}/q) in {time.time()-t0:.1f}s')

        print('Extracting features...')
        t0 = time.time()
        X = extractor.extract_batch_from_cache(CACHE_ROOT, SPLIT, all_q, all_t, all_scores)
        print(f'Features: {X.shape} in {time.time()-t0:.1f}s')

        val_probs = ranker.predict_proba(X)

        lo_k = start_q * n_targets
        hi_k = (start_q + n_val) * n_targets
        sub = pos_keys[np.searchsorted(pos_keys, lo_k, 'left'):np.searchsorted(pos_keys, hi_k, 'right')]
        val_gt = {q: set() for q in val_queries}
        for pk in sub:
            q = int(pk // n_targets)
            t = int(pk % n_targets)
            if q in val_gt:
                val_gt[q].add(t)

        t_sing, t_match, best_f05 = optimize_thresholds_aggressive(all_q, all_t, val_probs, val_gt, val_queries)

        with open(POLICY_PATH, 'w') as f:
            json.dump({'t_singleton': t_sing, 't_match': t_match, 'val_f05': best_f05}, f, indent=2)
        print(f'Saved to {POLICY_PATH}')

    run_full_inference(t_sing, t_match, top_k=args.top_k)
    print('\n[DONE] Submit output/matching_results.tsv and output/candidate_pairs.tsv')


if __name__ == '__main__':
    main()

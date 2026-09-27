import argparse
import os
import sys
from collections import defaultdict
from zlib import crc32

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data import fastnorm as fn
from src.data.loader import fld

CACHE = r"S:\Amazon_ML_Challenge\cache"
SPLIT = "train"

_STREET_TYPES = {
    "st", "rd", "ave", "blvd", "dr", "ln", "hwy", "pkwy", "ct", "cir", "pl",
    "plz", "sq", "ter", "trl", "way", "aly", "apt", "ste", "fl", "bldg",
    "blk", "twr", "rue", "allee", "chemin", "imp", "quai", "cours", "rond",
    "esp", "fbg", "opp", "near", "beside", "behind", "n", "s", "e", "w",
    "ne", "nw", "se", "sw", "h", "no", "plot", "shop", "flat", "gali", "room",
    "floor", "sector", "sec", "flr", "unit", "in", "at", "to", "by", "of",
    "and", "the", "dl", "ma", "tx", "ca", "ny", "il", "pa", "oh", "ga", "nc",
}

SEED_FULL_NAME = 0x11
SEED_SORTED_NAME = 0x12
SEED_NAME_ADDRNO = 0x13
SEED_NAME_POSTAL = 0x14
SEED_ADDRNO_POSTAL = 0x15
SEED_ADDRNO_STREET = 0x16
SEED_PARTIAL_NAME = 0x17
SEED_POSTAL_NAME1 = 0x18
SEED_NAME_FIRST2 = 0x19
SEED_NAME1_ADDRNO = 0x20
SEED_NAME1_STREET = 0x21
SEED_NAME1_POSTAL = 0x22
SEED_ADDR_TOK_PAIR = 0x23
SEED_PREFIX4_POSTAL = 0x24
SEED_PREFIX4_ADDRNO = 0x25

FAMILY_WEIGHT = {
    SEED_FULL_NAME: 16.0,
    SEED_SORTED_NAME: 14.0,
    SEED_NAME_FIRST2: 13.5,
    SEED_NAME_ADDRNO: 13.0,
    SEED_NAME1_ADDRNO: 12.5,
    SEED_NAME_POSTAL: 12.0,
    SEED_NAME1_POSTAL: 11.5,
    SEED_NAME1_STREET: 11.0,
    SEED_PARTIAL_NAME: 11.0,
    SEED_PREFIX4_POSTAL: 10.5,
    SEED_PREFIX4_ADDRNO: 10.5,
    SEED_ADDRNO_STREET: 10.5,
    SEED_ADDRNO_POSTAL: 10.0,
    SEED_POSTAL_NAME1: 10.0,
    SEED_ADDR_TOK_PAIR: 9.5,
}

MAX_KEYS_PER_DOC = 14

def advanced_doc_keys(name_norm: str, addr_norm: str):
    if not name_norm and not addr_norm:
        return []

    core = [t for t in name_norm.split() if t and t not in fn.LEGAL_DROP]
    addr_toks = addr_norm.split() if addr_norm else []
    digits = [t for t in addr_toks if t.isdigit()]

    addr_no = ""
    for t in addr_toks:
        if t.isdigit() and 1 <= len(t) <= 5:
            addr_no = t
            break
    if not addr_no and digits:
        addr_no = digits[0]

    postal = ""
    for t in reversed(digits):
        if len(t) in (5, 6):
            postal = t
            break

    distinctive_addr = [t for t in addr_toks if not t.isdigit() and t not in _STREET_TYPES and len(t) >= 3]

    keys = []
    def add(seed: int, text: str):
        keys.append((crc32(text.encode("utf-8"), seed), FAMILY_WEIGHT.get(seed, 10.0)))

    if core:
        full = " ".join(core)
        add(SEED_FULL_NAME, full)
        add(SEED_SORTED_NAME, " ".join(sorted(core)))
        if len(core) >= 2:
            add(SEED_NAME_FIRST2, core[0] + " " + core[1])
        if len(core) >= 3:
            add(SEED_PARTIAL_NAME, " ".join(sorted(core)[:3]))
        
        name1 = core[0]
        if addr_no:
            add(SEED_NAME_ADDRNO, full + "|" + addr_no)
            add(SEED_NAME1_ADDRNO, name1 + "|" + addr_no)
            if len(name1) >= 4:
                add(SEED_PREFIX4_ADDRNO, name1[:4] + "|" + addr_no)
        if postal:
            add(SEED_NAME_POSTAL, full + "|" + postal)
            add(SEED_POSTAL_NAME1, postal + "|" + name1)
            add(SEED_NAME1_POSTAL, name1 + "|" + postal)
            if len(name1) >= 4:
                add(SEED_PREFIX4_POSTAL, name1[:4] + "|" + postal)
        if distinctive_addr:
            for st in distinctive_addr[:2]:
                add(SEED_NAME1_STREET, name1 + "|" + st)

    if addr_no and postal:
        add(SEED_ADDRNO_POSTAL, addr_no + "|" + postal)
    
    if addr_no and distinctive_addr:
        for st in distinctive_addr[:3]:
            add(SEED_ADDRNO_STREET, addr_no + "|" + st)

    if len(distinctive_addr) >= 2:
        for i in range(min(len(distinctive_addr)-1, 2)):
            add(SEED_ADDR_TOK_PAIR, distinctive_addr[i] + "|" + distinctive_addr[i+1])

    best = {}
    order = []
    for k, w in keys:
        if k not in best:
            best[k] = w
            order.append(k)
        elif w > best[k]:
            best[k] = w
    return [(k, best[k]) for k in order][:MAX_KEYS_PER_DOC]

def main():
    n_target = 120_000
    n_query = 4_000
    seed = 7

    s2 = os.path.join(CACHE, SPLIT, "s2")
    s3 = os.path.join(CACHE, SPLIT, "s3")
    s1 = os.path.join(CACHE, SPLIT, "s1")
    gt = os.path.join(CACHE, SPLIT, "gt")

    n2 = len(np.load(fld(s2, "idh"), mmap_mode="r"))
    n3 = len(np.load(fld(s3, "idh"), mmap_mode="r"))
    n_targets = n2 + n3
    n_s1 = len(np.load(fld(s1, "idh"), mmap_mode="r"))
    pos_keys = np.load(fld(gt, "pos_keys"))

    rng = np.random.default_rng(seed)
    per = n_target // 2
    tgt_rows = np.concatenate([np.arange(0, per), n2 + np.arange(0, per)])
    in_subsample = np.zeros(n_targets, dtype=bool)
    in_subsample[tgt_rows] = True

    keymap = defaultdict(list)
    names = {"s2": np.load(fld(s2, "name"), mmap_mode="r"), "s3": np.load(fld(s3, "name"), mmap_mode="r")}
    addrs = {"s2": np.load(fld(s2, "addr"), mmap_mode="r"), "s3": np.load(fld(s3, "addr"), mmap_mode="r")}
    n_nonzero = 0
    for tag, base, off in (("s2", s2, 0), ("s3", s3, n2)):
        nm = names[tag]
        ad = addrs[tag]
        local = tgt_rows[tgt_rows >= off][:per] - off
        for r in local.tolist():
            ks = advanced_doc_keys(nm[r].decode("utf-8", "ignore"), ad[r].decode("utf-8", "ignore"))
            if ks:
                n_nonzero += 1
            for k, _ in ks:
                keymap[k].append(r + off)
    print(f"target subsample: {len(tgt_rows):,} rows, {n_nonzero:,} produced >=1 key, {len(keymap):,} distinct keys")
    avg_post = np.mean([len(v) for v in keymap.values()])
    print(f"avg posting list length: {avg_post:.1f}, max: {max(len(v) for v in keymap.values())}")

    query_rows = np.sort(rng.choice(n_s1, size=n_query, replace=False))
    lo_k, hi_k = int(query_rows[0]) * n_targets, int(query_rows[-1]) * n_targets
    sl = pos_keys[np.searchsorted(pos_keys, lo_k, "left"):np.searchsorted(pos_keys, hi_k, "right")]
    sr = (sl // n_targets).astype(np.int64)
    sel = np.isin(sr, query_rows)
    sl = sl[sel]
    sr = sr[sel]
    sc = (sl % n_targets).astype(np.int64)

    truth = defaultdict(set)
    for r, c in zip(sr.tolist(), sc.tolist()):
        if in_subsample[c]:
            truth[r].add(c)
    total_in = sum(len(v) for v in truth.values())
    print(f"queries: {len(query_rows):,}; true pairs in subsample: {total_in:,} (from {len(truth):,} queries)")

    nm1 = np.load(fld(s1, "name"), mmap_mode="r")
    ad1 = np.load(fld(s1, "addr"), mmap_mode="r")
    hit = 0
    n_cand = 0
    per_q = []
    for r in query_rows.tolist():
        ks = advanced_doc_keys(nm1[r].decode("utf-8", "ignore"), ad1[r].decode("utf-8", "ignore"))
        cands = set()
        for k, _ in ks:
            cands.update(keymap.get(k, ()))
        n_cand += len(cands)
        t = truth.get(r)
        if t:
            h = len(cands & t)
            hit += h
            per_q.append(h / len(t))
    per_q = np.array(per_q) if per_q else np.zeros(1)
    print()
    print("=" * 60)
    print(f"  ADVANCED COMPOSITE-KEY RECALL (subsample) : {hit / max(1, total_in):.4f}")
    print(f"  mean per-query recall                      : {per_q.mean():.4f}")
    print(f"  full-recall queries                        : {(per_q == 1.0).mean():.1%}")
    print(f"  avg candidates per query                   : {n_cand / len(query_rows):.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

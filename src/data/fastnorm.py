"""
Fast, compiled-once text normalization for large-scale entity resolution.

Every regex / translation table here is built at import time so that the hot
loops (millions of records) only pay C-level string scanning.

Design goals
------------
1. Deterministic: the same input always yields the same output, because the
   normalized strings are baked into an on-disk cache and reused everywhere.
2. Country-agnostic: no hard-coded filter on ``country``. Legal-form tokens for
   the US, India and France are canonicalized, but an unseen label still
   normalizes cleanly.
3. Loss-aware: digits survive normalization, because house numbers and postal
   codes are strong entity-resolution signals.
"""

import re
import unicodedata
from typing import List, Set, Tuple

# --------------------------------------------------------------------------
# Accent folding
# --------------------------------------------------------------------------
# A precomputed translation table is far faster than calling
# unicodedata.normalize() per character. We fold the full Latin-1 Supplement
# and Latin Extended-A ranges, which covers US / India / France / transliterated
# Latin business names.
_ACCENT_FOLD = {}


def _build_accent_table() -> dict:
    table = {}
    for cp in range(0x00C0, 0x0180):
        ch = chr(cp)
        decomposed = unicodedata.normalize("NFD", ch)
        stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        if stripped and stripped != ch:
            table[ch] = stripped
    # Ligatures and typographic characters that NFD does not decompose.
    table.update({
        "\u00df": "ss",   # sharp s
        "\u00fe": "tf",   # small ligature thorn
        "\u0153": "oe",   # oe ligature
        "\u0152": "OE",
        "\u00c6": "AE", "\u00e6": "ae",
        "\u00d0": "D", "\u00f0": "d",
        "\u00d8": "O", "\u00f8": "o",
    })
    return table


_ACCENT_FOLD = _build_accent_table()
_ACCENT_TABLE = str.maketrans(_ACCENT_FOLD)

# Anything that is not [a-z0-9] becomes a single space. Built once.
_NON_ALNUM_TABLE = {i: " " for i in range(128) if chr(i) not in
                    "abcdefghijklmnopqrstuvwxyz0123456789"}
# ASCII control / punctuation space
_NON_ALNUM_TABLE[9] = " "
_NON_ALNUM_TABLE[10] = " "
_NON_ALNUM_TABLE[13] = " "
_NON_ALNUM_TABLE[32] = " "

# Apostrophes, quote marks and full stops are *deleted* rather than spaced out.
# "ORELEE'S" -> "orelees" (not "orelee s") and "S.A.R.L." -> "sarl" (not
# "s a r l"), which is what actually matches across sources that spell the
# token differently.
_DELETE_CHARS = "'`\u2018\u2019\u201c\u201d\u00b7\u2022."
for _c in _DELETE_CHARS:
    _NON_ALNUM_TABLE[ord(_c)] = ""


def _strip_non_ascii_chars() -> None:
    """Extend the non-alphanumeric table across the whole BMP + common symbols.

    Doing this up front keeps the hot loop free of per-character branches.
    Characters listed in ``_DELETE_CHARS`` are removed outright rather than
    replaced by a space.
    """
    delete = {ord(c) for c in _DELETE_CHARS}
    for cp in range(128, 0x2FFF):
        if cp in delete:
            _NON_ALNUM_TABLE[cp] = ""
        elif chr(cp) not in "abcdefghijklmnopqrstuvwxyz0123456789":
            _NON_ALNUM_TABLE[cp] = " "


_strip_non_ascii_chars()

_WS_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")

# --------------------------------------------------------------------------
# Legal-form vocabulary (country-agnostic, open set safe)
# --------------------------------------------------------------------------
# Single-token canonicalization. Keys are matched on the accent-folded,
# punctuation-stripped, lower-cased token.
LEGAL_CANON = {
    # United States / generic anglophone
    "corp": "inc", "corporation": "inc", "inc": "inc", "incorporated": "inc",
    "llc": "llc", "l.l.c": "llc", "ltd": "ltd", "limited": "ltd",
    "co": "co", "company": "co", "plc": "plc", "pc": "pc", "pa": "pa",
    "lp": "lp", "llp": "llp", "lp.": "lp",
    # India
    "pvt": "pvt", "pvt.": "pvt", "private": "pvt",
    "ltd.": "ltd", "pvtltd": "pvtltd",
    # France / EU
    "sarl": "sarl", "s.a.r.l": "sarl", "sas": "sas", "s.a.s": "sas",
    "eurl": "eurl", "sci": "sci", "sca": "sca", "snc": "snc",
    "sa": "sa", "s.a": "sa", "sasu": "sas", "sas": "sas",
    "sprl": "sprl", "bv": "bv", "nv": "nv", "gmbh": "gmbh", "ag": "ag",
    # Generic filler that carries no discriminating signal
    "the": "", "and": "", "of": "", "&": "",
}

# Tokens dropped when building the legal-form-free variant of a name.
LEGAL_DROP = {
    "inc", "llc", "ltd", "co", "plc", "pc", "lp", "llp", "pvt", "pvtltd",
    "sarl", "sas", "eurl", "sci", "sca", "snc", "sa", "sas", "sprl", "bv",
    "nv", "gmbh", "ag", "the", "and", "of",
}

# Multi-word legal forms handled before tokenization, longest first.
_LEGAL_PHRASES = [
    ("private limited", "pvt ltd"),
    ("pvt ltd", "pvt ltd"),
    ("pvt. ltd.", "pvt ltd"),
    ("limited liability company", "llc"),
    ("limited liability partnership", "llp"),
    ("societe par actions simplifiee", "sas"),
    ("societe a responsabilite limitee", "sarl"),
    ("entreprise unipersonnelle a responsabilite limitee", "eurl"),
    ("societe civile immobiliere", "sci"),
    ("societe anonyme", "sa"),
]
_LEGAL_PHRASE_RE = [(re.compile(re.escape(a) + r"\s+"), b) for a, b in _LEGAL_PHRASES]

# --------------------------------------------------------------------------
# Address vocabulary
# --------------------------------------------------------------------------
ADDRESS_CANON = {
    # US / India street types
    "street": "st", "str": "st",
    "road": "rd", "rd.": "rd",
    "avenue": "ave", "ave.": "ave", "av": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "court": "ct", "ct.": "ct", "circle": "cir", "cir": "cir",
    "place": "pl", "plz": "plz", "plaza": "plz",
    "terrace": "ter", "square": "sq", "sq": "sq", "sqr": "sq",
    "trail": "trl", "way": "way", "alley": "aly", "aly": "aly",
    # Unit designators
    "apartment": "apt", "apt": "apt", "suite": "ste", "ste": "ste",
    "floor": "fl", "flr": "fl", "fl.": "fl",
    "building": "bldg", "bldg": "bldg", "bldg.": "bldg",
    "block": "blk", "blk": "blk", "tower": "twr",
    # India landmark markers
    "opposite": "opp", "opp": "opp", "opp.": "opp",
    "near": "near", "nr": "near", "beside": "beside", "besides": "beside",
    "adjacent": "beside", "behind": "behind", "bhd": "behind",
    "alongside": "beside",
    # French street types
    "rue": "rue", "r": "rue", "allee": "allee", "all": "allee", "chemin": "chemin",
    "ch": "chemin", "impasse": "imp", "imp": "imp",
    "quai": "quai", "cours": "cours", "rond": "rond",
    "esplanade": "esp", "faubourg": "fbg",
    # Generic filler inside addresses
    "the": "", "and": "", "of": "", "de": "", "du": "", "des": "",
    "la": "", "le": "", "les": "", "a": "", "en": "",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}

# Address tokens carrying no signal; dropped from the address token set.
ADDRESS_STOP = {"", "n", "s", "e", "w", "ne", "nw", "se", "sw"}


def base_clean(text: str) -> str:
    """Lowercase, fold accents, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    # Fast path: skip the accent pass entirely for pure-ASCII strings.
    if not t.isascii():
        t = t.translate(_ACCENT_TABLE)
    t = t.translate(_NON_ALNUM_TABLE)
    if "  " in t:
        t = _WS_RE.sub(" ", t)
    return t.strip()


def _canonicalize_tokens(tokens: List[str], table: dict) -> List[str]:
    out = []
    for tok in tokens:
        repl = table.get(tok)
        if repl is None:
            out.append(tok)
        elif repl:
            out.append(repl)
        # else: mapped to "" -> intentionally dropped
    return out


def normalize_name(name: str) -> str:
    """Canonicalize a business name, keeping one legal-form token."""
    cleaned = base_clean(name)
    if not cleaned:
        return ""
    for pattern, repl in _LEGAL_PHRASE_RE:
        if pattern.search(cleaned):
            cleaned = pattern.sub(repl, cleaned)
    toks = _canonicalize_tokens(cleaned.split(" "), LEGAL_CANON)
    return " ".join(toks)


def normalize_name_core(name: str) -> str:
    """Canonicalize a business name with all legal-form tokens removed."""
    cleaned = base_clean(name)
    if not cleaned:
        return ""
    for pattern, repl in _LEGAL_PHRASE_RE:
        if pattern.search(cleaned):
            cleaned = pattern.sub(repl, cleaned)
    toks = [t for t in cleaned.split(" ") if t and t not in LEGAL_DROP]
    return " ".join(toks)


def normalize_addr(address: str) -> str:
    """Canonicalize a business address."""
    cleaned = base_clean(address)
    if not cleaned:
        return ""
    toks = _canonicalize_tokens(cleaned.split(" "), ADDRESS_CANON)
    return " ".join(toks)


def name_tokens(name_norm: str) -> List[str]:
    """Content tokens of an already-normalized name (legal form removed)."""
    return [t for t in name_norm.split(" ") if t and t not in LEGAL_DROP]


def addr_tokens(addr_norm: str) -> List[str]:
    """Content tokens of an already-normalized address (stopwords removed)."""
    return [t for t in addr_norm.split(" ") if t and t not in ADDRESS_STOP]


def all_numbers(text_norm: str) -> List[str]:
    """Digit runs in a normalized string, longest-first order preserved."""
    return _DIGITS_RE.findall(text_norm)


def postal_codes(text_norm: str) -> List[str]:
    """Postal-code-shaped digit runs (US/FR 5-digit, IN 6-digit)."""
    return re.findall(r"\d{5}(?:\d{1})?", text_norm)


def char_ngrams(text: str, n: int = 3) -> List[str]:
    """Character n-grams; the workhorse of typo-robust matching."""
    if not text:
        return []
    if len(text) <= n:
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def composite(name_norm: str, addr_norm: str) -> str:
    """Single string used for coarse lexical retrieval."""
    return (name_norm + " " + addr_norm).strip()

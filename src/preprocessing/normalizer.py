"""
Multi-Country Text & Address Normalization Module
Handles noise patterns across US, India, and France.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple


class TextNormalizer:
    def __init__(self):
        # Legal suffix mapping across US, India, and France
        self.legal_suffixes: Dict[str, str] = {
            # US / Generic
            r"\b(corporation|corp|inc|incorporated)\b": "inc",
            r"\b(limited liability company|llc|l\.l\.c\.)\b": "llc",
            r"\b(limited liability partnership|llp|l\.l\.p\.)\b": "llp",
            r"\b(company|co)\b": "co",
            r"\b(limited|ltd)\b": "ltd",
            # India specific
            r"\b(private limited|pvt ltd|pvt\. ltd\.|private ltd|p\. ltd|pvt)\b": "pvt ltd",
            r"\b(public limited|pub ltd)\b": "ltd",
            # France specific
            r"\b(societe a responsabilite limitee|sarl|s\.a\.r\.l\.)\b": "sarl",
            r"\b(societe par actions simplifiee|sas|s\.a\.s\.)\b": "sas",
            r"\b(societe anonyme|sa|s\.a\.)\b": "sa",
            r"\b(entreprise unipersonnelle a responsabilite limitee|eurl|e\.u\.r\.l\.)\b": "eurl",
            r"\b(societe civile immobiliere|sci|s\.c\.i\.)\b": "sci",
        }

        # Address abbreviation mappings
        self.address_replacements: Dict[str, str] = {
            r"\b(street|str|st\.)\b": "st",
            r"\b(road|rd\.)\b": "rd",
            r"\b(avenue|ave\.)\b": "ave",
            r"\b(boulevard|blvd\.)\b": "blvd",
            r"\b(drive|dr\.)\b": "dr",
            r"\b(lane|ln\.)\b": "ln",
            r"\b(highway|hwy\.)\b": "hwy",
            r"\b(parkway|pkwy\.)\b": "pkwy",
            r"\b(apartment|apt|apt\.)\b": "apt",
            r"\b(suite|ste|ste\.)\b": "ste",
            r"\b(floor|fl|flr)\b": "fl",
            r"\b(building|bldg)\b": "bldg",
            r"\b(number|no|no\.)\b": "#",
            # Landmark abbreviations (frequent in India)
            r"\b(opposite|opp|opp\.)\b": "opp",
            r"\b(near|nr|nr\.)\b": "near",
            r"\b(behind|bhnd)\b": "behind",
            r"\b(beside|adj|adjacent)\b": "beside",
            # French address terms
            r"\b(rue|r\.)\b": "rue",
            r"\b(avenue|av\.)\b": "ave",
            r"\b(boulevard|bd)\b": "blvd",
            r"\b(allee|all\.)\b": "allee",
            r"\b(place|pl\.)\b": "pl",
            r"\b(chemin|ch\.)\b": "chemin",
        }

    def strip_accents(self, text: str) -> str:
        """Removes diacritics / accents (e.g. for French text: café -> cafe)."""
        if not text:
            return ""
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def clean_text_basic(self, text: str) -> str:
        """Standard basic text cleaning."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self.strip_accents(text.lower())
        # Replace '&' with 'and'
        text = re.sub(r"&", " and ", text)
        # Remove special characters except alphanumeric, hash, and whitespace
        text = re.sub(r"[^\w\s#\-]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_business_name(self, name: str, remove_legal: bool = False) -> str:
        """
        Normalizes business entity names.
        If remove_legal=True, strips legal suffixes for core brand matching.
        """
        cleaned = self.clean_text_basic(name)
        if not cleaned:
            return ""

        for pattern, replacement in self.legal_suffixes.items():
            if remove_legal:
                cleaned = re.sub(pattern, "", cleaned)
            else:
                cleaned = re.sub(pattern, replacement, cleaned)

        return re.sub(r"\s+", " ", cleaned).strip()

    def normalize_address(self, address: str) -> str:
        """Normalizes business address components."""
        cleaned = self.clean_text_basic(address)
        if not cleaned:
            return ""

        for pattern, replacement in self.address_replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned)

        return re.sub(r"\s+", " ", cleaned).strip()

    def extract_numbers(self, text: str) -> Set[str]:
        """Extracts numeric tokens (house numbers, postal codes, suite numbers)."""
        if not text:
            return set()
        return set(re.findall(r"\b\d+\b", text))

    def extract_postal_code(self, address: str, country: str = "") -> str:
        """Extracts postal codes (US 5-digit, India 6-digit PIN, France 5-digit)."""
        if not address:
            return ""
        # Match 6-digit for India, 5-digit for US / France
        match_6 = re.findall(r"\b\d{6}\b", address)
        if match_6:
            return match_6[-1]
        match_5 = re.findall(r"\b\d{5}\b", address)
        if match_5:
            return match_5[-1]
        return ""

    def create_composite_representation(self, name: str, address: str, country: str) -> str:
        """Creates a standardized unified string representation for embeddings."""
        norm_name = self.normalize_business_name(name)
        norm_addr = self.normalize_address(address)
        norm_country = (country or "").strip().lower()
        return f"{norm_name} [SEP] {norm_addr} [SEP] {norm_country}"


# Global normalizer instance
normalizer = TextNormalizer()

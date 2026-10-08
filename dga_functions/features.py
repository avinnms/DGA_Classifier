import math
import re
from collections import Counter
import tldextract

try:
    from machine import entropy
except ImportError:
    def entropy(s):
        count = Counter(s)
        return -sum(c/len(s) * math.log2(c/len(s)) for c in count.values())

_extract = tldextract.TLDExtract(include_psl_private_domains=True, suffix_list_urls=())
vowels = set("aeiou")

def label_of(domain: str) -> str:
    """strips the domain name from an address"""
    ext = _extract(domain.strip().lower().rstrip("."))
    return ext.domain or domain

def longest(s, charset):
    best = cur = 0
    for i in s:
        cur = cur + 1 if i in charset else 0
        best = max(best, cur)
    return best

def handmade(label: str) -> dict:
    n = len(label) or 1
    letters = [c for c in label if c.isalpha()]
    consonants = set("bcdfghjklmnpqrstvwxyz")
    return {
        "length": len (label),
        "entropy": entropy(label) if label else 0.0,
        "digit_ratio": sum(c.isdigit() for c in label) / n,
        "vowel_ratio": sum(c in vowels for c in letters) / (len(letters) or 1),
        "count_hyphen": label.count("-"),
        "max_consonant": longest(label, consonants),
        "max_digit": longest(label, set("0123456789")),
        "unique_char_ratio": len(set(label)) / n,
    }

FEATURE_NAMES = list(handmade("example").keys())

"""Tokenizing and stemming, shared by the indexer and (in Milestone 2) queries."""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer

# A token is a run of ASCII letters/digits so Self-Driving turns into Self, Driving.
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

_stemmer = PorterStemmer()


# Porter stem one word. Cached since words repeat a lot.
@lru_cache(maxsize=None)
def stem(word: str) -> str:
    return _stemmer.stem(word)


# Split text into lowercase alphanumeric tokens and stem each one.
def tokenize(text: str) -> list[str]:
    return [stem(match.lower()) for match in _TOKEN_PATTERN.findall(text)]

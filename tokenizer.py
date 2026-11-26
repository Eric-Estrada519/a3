"""Tokenizing and stemming, shared by the indexer and queries."""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer

# A token is a run of ASCII letters/digits so Self-Driving turns into Self, Driving.
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

# A dotted abbreviation: two or more groups of 1-2 letters joined by periods, like
# Ph.D, U.S or U.S.A. It must not touch another letter, digit, or period on either side,
# so web addresses (www.ics.uci.edu) and decimals (3.14) do not count.
_ABBREVIATION_PATTERN = re.compile(r"(?<![a-zA-Z0-9.])[a-zA-Z]{1,2}(?:\.[a-zA-Z]{1,2})+(?![a-zA-Z0-9])")

_stemmer = PorterStemmer()


# Porter stem one word. Cached since words repeat a lot.
@lru_cache(maxsize=None)
def stem(word: str) -> str:
    return _stemmer.stem(word)


# Split text into lowercase alphanumeric tokens and stem each one. A dotted abbreviation
# like "Ph.D." gives its pieces ("ph", "d") and then the joined word ("phd"), so the
# query "phd" can find pages that write "Ph.D.".
def tokenize(text: str) -> list[str]:
    tokens = []
    position = 0
    for abbreviation in _ABBREVIATION_PATTERN.finditer(text):
        # Ordinary tokens between the previous abbreviation and this one.
        tokens += _stemmed_words(text[position:abbreviation.start()])
        pieces = abbreviation.group().split(".")
        tokens += _stemmed_words(" ".join(pieces))
        tokens += _stemmed_words("".join(pieces))
        position = abbreviation.end()
    tokens += _stemmed_words(text[position:])
    return tokens


# The plain rule: every alphanumeric run, lowercased and stemmed.
def _stemmed_words(text: str) -> list[str]:
    return [stem(word.lower()) for word in _TOKEN_PATTERN.findall(text)]


# Phrase lengths saved by the indexer and matched by the search: 2-grams and 3-grams.
NGRAM_SIZES = (2, 3)


# Join each run of n neighboring tokens: with n=2, ["ic", "31", "cours"] -> ["ic 31",
# "31 cours"]. Used for 2-gram and 3-gram (phrase) matching so word order is not lost.
def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

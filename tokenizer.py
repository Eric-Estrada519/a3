import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    return [stem(match.lower()) for match in _TOKEN_PATTERN.findall(text)]

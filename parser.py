import json
import warnings
from pathlib import Path
from urllib.parse import urldefrag

from bs4 import BeautifulSoup, Comment, NavigableString, XMLParsedAsHTMLWarning

from tokenizer import tokenize

IMPORTANT_TAGS = frozenset({"title", "h1", "h2", "h3", "b", "strong"})

REMOVED_TAGS = ("script", "style", "noscript")


class PageError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def defragment(url: str) -> str:
    return urldefrag(url).url


def _decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    fallback = raw.decode("utf-8-sig", errors="replace")
    try:
        declared = json.loads(fallback).get("encoding")
    except (json.JSONDecodeError, AttributeError):
        return fallback
    if not isinstance(declared, str):
        return fallback
    try:
        redecoded = raw.decode(declared, errors="replace")
    except LookupError:
        return fallback
    try:
        json.loads(redecoded)
    except json.JSONDecodeError:
        return fallback
    return redecoded


def load_page(path) -> tuple[str, str]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise PageError(f"read error: {exc}") from exc
    text = _decode_bytes(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PageError("invalid JSON") from exc
    if not isinstance(data, dict):
        raise PageError("invalid JSON")

    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        raise PageError("missing url")
    content = data.get("content")
    if not isinstance(content, str) or not content:
        raise PageError("missing content")
    return url, content


def extract_term_counts(html: str) -> dict[str, tuple[int, int]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        raise PageError(f"parse error: {exc}") from exc

    for tag in soup.find_all(REMOVED_TAGS):
        if not tag.decomposed:
            tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    counts: dict[str, list[int]] = {}
    for node in soup.find_all(string=True):
        if type(node) is not NavigableString:
            continue
        tokens = tokenize(str(node))
        if not tokens:
            continue
        important = any(parent.name in IMPORTANT_TAGS for parent in node.parents)
        for token in tokens:
            entry = counts.setdefault(token, [0, 0])
            entry[0] += 1
            if important:
                entry[1] += 1

    return {term: (tf, important_tf) for term, (tf, important_tf) in counts.items()}

"""Load crawled JSON pages and turn their HTML into per-term counts."""
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urljoin

from bs4 import BeautifulSoup, Comment, NavigableString, XMLParsedAsHTMLWarning

from tokenizer import NGRAM_SIZES, ngrams, tokenize

# Important text is split into three kinds so search can weight each one differently.
TITLE_TAGS = frozenset({"title"})
HEADING_TAGS = frozenset({"h1", "h2", "h3"})
BOLD_TAGS = frozenset({"b", "strong"})

# These tags hold code, not page text, so they are deleted first.
REMOVED_TAGS = ("script", "style", "noscript")


# Raised for a page that must be skipped. reason goes in the skip log.
class PageError(Exception):
    # Store the short reason for the skip log.
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Remove the #fragment part of a URL.
def defragment(url: str) -> str:
    return urldefrag(url).url


# Decode file bytes without ever failing. this does utf first then the declared encoding
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
    except LookupError:  # unknown codec name
        return fallback
    # Only trust the declared encoding if the result still parses as JSON.
    try:
        json.loads(redecoded)
    except json.JSONDecodeError:
        return fallback
    return redecoded


# Read one JSON file and return (url, content), or raise PageError.
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


# Say which kind of text a text node is: "title", "heading", "bold", or None for body
# text. If it sits inside several kinds (like <h1><b>x</b></h1>), the strongest wins:
# title, then heading, then bold. So each word is counted in at most one kind.
def _text_kind(node) -> str | None:
    parent_names = {parent.name for parent in node.parents}
    if parent_names & TITLE_TAGS:
        return "title"
    if parent_names & HEADING_TAGS:
        return "heading"
    if parent_names & BOLD_TAGS:
        return "bold"
    return None


# Position of each kind's count in a page's (tf, title_tf, heading_tf, bold_tf) tuple.
_KIND_SLOT = {"title": 1, "heading": 2, "bold": 3}


# Everything the indexer needs from one page.
@dataclass
class ParsedPage:
    counts: dict[str, tuple[int, int, int, int]]  # term -> (tf, title_tf, heading_tf, bold_tf)
    ngram_counts: dict[str, int]                 # 2- or 3-word phrase -> times it appears
    links: list[tuple[str, list[str]]]            # (target URL, anchor text tokens)
    fingerprint: str                              # hash of the page's words, for duplicates


# Return just {term: (tf, title_tf, heading_tf, bold_tf)} for one page.
def extract_term_counts(html: str) -> dict[str, tuple[int, int, int, int]]:
    return parse_page(html, "").counts


# Parse one page (its URL is needed to resolve relative links). tf counts every
# occurrence. title_tf, heading_tf and bold_tf count only occurrences inside that kind
# of tag, so together they never exceed tf. Each text node is tokenized on its own so
# words in neighboring tags never merge into one word.
def parse_page(html: str, page_url: str) -> ParsedPage:
    try:
        # RSS/XML pages parse fine as HTML. Hides bs4's warning about them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        raise PageError(f"parse error: {exc}") from exc

    for tag in soup.find_all(REMOVED_TAGS):
        # A tag nested in an already removed tag is gone.
        if not tag.decomposed:
            tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    counts: dict[str, list[int]] = {}  # term -> [tf, title_tf, heading_tf, bold_tf]
    page_tokens: list[str] = []        # every token of the page, in order
    for node in soup.find_all(string=True):
        # Only plain text nodes.
        if type(node) is not NavigableString:
            continue
        tokens = tokenize(str(node))
        if not tokens:
            continue
        page_tokens += tokens
        kind = _text_kind(node)
        for token in tokens:
            entry = counts.setdefault(token, [0, 0, 0, 0])
            entry[0] += 1
            if kind:
                entry[_KIND_SLOT[kind]] += 1

    # 2- and 3-word phrases are taken over the whole page in order, so "ICS <b>31</b>"
    # still gives the phrase "ic 31" even though it is two text nodes.
    ngram_counts: dict[str, int] = {}
    for n in NGRAM_SIZES:
        for phrase in ngrams(page_tokens, n):
            ngram_counts[phrase] = ngram_counts.get(phrase, 0) + 1

    # Two pages with exactly the same words in the same order get the same hash.
    fingerprint = hashlib.sha256(" ".join(page_tokens).encode("utf-8")).hexdigest()

    return ParsedPage(
        counts={term: tuple(entry) for term, entry in counts.items()},
        ngram_counts=ngram_counts,
        links=_extract_links(soup, page_url),
        fingerprint=fingerprint,
    )


# Return (target URL, anchor text tokens) for every <a href> that has some words.
# The target URL is made absolute and its #fragment removed so it can be matched
# against the indexed URLs. A link whose href cannot be parsed is skipped.
def _extract_links(soup, page_url: str) -> list[tuple[str, list[str]]]:
    links = []
    for anchor in soup.find_all("a", href=True):
        words = tokenize(anchor.get_text(" "))
        if not words:
            continue
        try:
            target = defragment(urljoin(page_url, anchor["href"]))
        except ValueError:
            continue
        links.append((target, words))
    return links

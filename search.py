"""Ranked search over the saved index. Console: python search.py [output_dir]"""
import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from indexer import NGRAMS_FILE, DOC_IDS_FILE, INDEX_FILE, PAGERANK_FILE
from tokenizer import NGRAM_SIZES, ngrams, tokenize

# Extra weight for a query word that appears in each kind of important text.
# A title is the best summary of a page, so it counts most. Kept in one place for tuning.
TITLE_WEIGHT = 3.0     # <title>
HEADING_WEIGHT = 2.0   # <h1>, <h2>, <h3>
BOLD_WEIGHT = 1.5      # <b>, <strong>
ANCHOR_WEIGHT = 2.0    # text of links from other pages that point to this page

# Weight of the bonus a page gets for containing two or three query words side by side,
# in the same order as the query (a 2-gram like "ics 31" or a 3-gram like "how to apply").
NGRAM_WEIGHT = 1.0

# A phrase found on more than this share of all pages is site boilerplate (menu or
# footer text such as "Graduate Admissions" or "Privacy Policy"), so it gets no bonus.
# Without this, short pages that just carry the menu beat the real page.
MAX_NGRAM_PAGE_SHARE = 0.25

# How strongly long pages are penalized. A page's score is divided by
# (vector length ** LENGTH_POWER). 1.0 is full cosine similarity, 0 is no penalty.
# Full cosine pushed useful long pages (the CS home page, the course list) out of the
# top results, so the square root (0.5) is used as a softer penalty.
LENGTH_POWER = 0.5

# How much PageRank (how well linked a page is) can raise a page's score. The page with
# the highest PageRank has its score multiplied by (1 + PAGERANK_WEIGHT), others by
# less. Multiplying (not adding) keeps the boost in proportion: it breaks near-ties
# between similar matches but cannot lift a weak match over a much better one.
# 0.3 lifts the menu pages people look for (cs/faculty/, cs/about-the-department/) to
# the top; at 0.4 and above, menu pages like the housing page crowd into unrelated queries.
PAGERANK_WEIGHT = 0.3

# Number of results the console shows per query.
TOP_K = 10


# One ranked search hit.
@dataclass
class Result:
    doc_id: int
    url: str
    score: float


# Holds the index in memory and answers queries against it.
class SearchEngine:
    # Keep the index (term -> postings), the doc ID -> URL map, and the phrase index
    # ("word1 word2" or "word1 word2 word3" -> postings), and work out every page's length divisor once here
    # so no query has to. pagerank is {doc_id: rank}; without it no page gets a link boost.
    def __init__(self, index: dict, doc_ids: dict, ngram_index: dict | None = None,
                 pagerank: dict | None = None):
        self.index = index
        self.ngram_index = ngram_index or {}
        self.urls = {int(k): v for k, v in doc_ids.items()}
        self.num_docs = len(self.urls)
        self.length_divisor = {
            doc_id: length ** LENGTH_POWER for doc_id, length in _doc_lengths(index).items()
        }
        self.link_boost = _link_boosts(pagerank or {}, self.num_docs)

    # Build an engine from the files the indexer saved in output_dir. An index saved
    # before PageRank was added has no PageRank file; it still loads, with no link boost.
    @classmethod
    def from_dir(cls, output_dir) -> "SearchEngine":
        output_dir = Path(output_dir)
        index = json.loads((output_dir / INDEX_FILE).read_text(encoding="utf-8"))
        doc_ids = json.loads((output_dir / DOC_IDS_FILE).read_text(encoding="utf-8"))
        ngram_index = json.loads((output_dir / NGRAMS_FILE).read_text(encoding="utf-8"))
        pagerank_path = output_dir / PAGERANK_FILE
        pagerank = (json.loads(pagerank_path.read_text(encoding="utf-8"))
                    if pagerank_path.is_file() else None)
        return cls(index, doc_ids, ngram_index, pagerank)

    # Return the top k pages for a query: more matched terms first, then higher score.
    # Score = sum over query terms of idf * term weight / length divisor, plus a bonus
    # for each 2- or 3-word run of query words that the page also has side by side.
    def search(self, query: str, k: int = 5) -> list[Result]:
        scores: dict[int, float] = {}
        matched: dict[int, int] = {}
        tokens = tokenize(query)
        for term in dict.fromkeys(tokens):  # unique terms, in order
            postings = self.index.get(term)
            if not postings:
                continue
            idf = math.log10(self.num_docs / len(postings))
            for posting in postings:
                doc_id = posting["doc_id"]
                # Dividing by the page's length stops long pages (like news archives)
                # from winning just because they contain more words.
                weight = idf * _term_weight(posting) / self.length_divisor[doc_id]
                scores[doc_id] = scores.get(doc_id, 0.0) + weight
                matched[doc_id] = matched.get(doc_id, 0) + 1
        # Phrase bonus: plain tf-idf ignores word order, so "ICS ... 31" far apart
        # scores like "ICS 31". Pages with the exact 2- or 3-word phrase get extra score,
        # so a page saying "how to apply" beats one with only "how to" and "to apply".
        # Such a page has all the phrase's words, so it is already in scores.
        phrases = [p for n in NGRAM_SIZES for p in ngrams(tokens, n)]
        for phrase in dict.fromkeys(phrases):
            postings = self.ngram_index.get(phrase)
            if not postings or len(postings) > MAX_NGRAM_PAGE_SHARE * self.num_docs:
                continue
            idf = math.log10(self.num_docs / len(postings))
            for posting in postings:
                doc_id = posting["doc_id"]
                bonus = NGRAM_WEIGHT * idf * _damp(posting["tf"]) / self.length_divisor[doc_id]
                scores[doc_id] += bonus
        # Link boost: a page that many pages link to (like a home page or the faculty
        # list) is more likely the page the user wants, so its score is raised a little.
        for doc_id in scores:
            scores[doc_id] *= self.link_boost.get(doc_id, 1.0)
        # Only the best k are needed, so pick them without sorting every match.
        top = heapq.nsmallest(k, scores, key=lambda d: (-matched[d], -scores[d], d))
        return [Result(d, self.urls[d], scores[d]) for d in top]


# Log-damped count: 0 stays 0, 1 -> 1, 10 -> 2, 100 -> 3. The 10th repeat of a word
# adds much less than the first, so a page can't win just by repeating a word.
def _damp(count: int) -> float:
    return 1 + math.log10(count) if count > 0 else 0.0


# Weight of one term in one page (before idf): damped tf plus a bonus for each kind of
# important text the term appears in, and for links from other pages that use it.
def _term_weight(posting: dict) -> float:
    return (_damp(posting["tf"])
            + TITLE_WEIGHT * _damp(posting["title_tf"])
            + HEADING_WEIGHT * _damp(posting["heading_tf"])
            + BOLD_WEIGHT * _damp(posting["bold_tf"])
            + ANCHOR_WEIGHT * _damp(posting["anchor_tf"]))


# Score multiplier for each page from its PageRank: 1 + PAGERANK_WEIGHT * (the page's
# log-scaled rank / the highest log-scaled rank), so the top page gets the full weight.
# PageRank is very uneven (a home page gets far more than one news story), so the log
# keeps a few heavily linked pages from taking the whole boost. rank * num_docs is 1
# for an average page.
def _link_boosts(pagerank: dict, num_docs: int) -> dict[int, float]:
    if not pagerank:
        return {}
    scaled = {int(doc_id): math.log1p(rank * num_docs) for doc_id, rank in pagerank.items()}
    top = max(scaled.values())
    return {doc_id: 1 + PAGERANK_WEIGHT * s / top for doc_id, s in scaled.items()}


# Length of each page's vector: sqrt of the sum of its squared term weights. Pages with
# many different words (or many repeats) get a longer vector, so a bigger divisor.
def _doc_lengths(index: dict) -> dict[int, float]:
    sums: dict[int, float] = {}
    for postings in index.values():
        for posting in postings:
            doc_id = posting["doc_id"]
            sums[doc_id] = sums.get(doc_id, 0.0) + _term_weight(posting) ** 2
    return {doc_id: math.sqrt(total) for doc_id, total in sums.items()}


# Console loop: read a query, print the ranked URLs and the time taken. An empty line
# just asks again. Type "quit" (or press Ctrl+C / Ctrl+D) to exit.
def run_console(engine: SearchEngine, k: int = TOP_K) -> None:
    while True:
        try:
            query = input("\nSearch (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in ("quit", "exit", "q"):
            return
        if not query:
            continue
        start = time.perf_counter()
        results = engine.search(query, k=k)
        ms = (time.perf_counter() - start) * 1000
        if not results:
            print(f"No results ({ms:.1f} ms)")
            continue
        print(f"Top {len(results)} results ({ms:.1f} ms):")
        for rank, r in enumerate(results, start=1):
            print(f"{rank:3}. {r.url}  (score {r.score:.3f})")


# Command line: load the index once, then answer queries until the user quits.
def main(argv=None) -> int:
    arg_parser = argparse.ArgumentParser(description="Search the index from the console.")
    arg_parser.add_argument("output_dir", nargs="?", default="output",
                            help="folder with the saved index (default: output)")
    args = arg_parser.parse_args(argv)

    print("Loading index...")
    start = time.perf_counter()
    engine = SearchEngine.from_dir(args.output_dir)
    print(f"Loaded {engine.num_docs} pages in {time.perf_counter() - start:.1f} s")
    run_console(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ranked search over the saved index. Console: python search.py [output_dir]"""
import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from indexer import DOC_IDS_FILE, INDEX_FILE
from tokenizer import tokenize

# Extra weight for a query word that appears in each kind of important text.
# A title is the best summary of a page, so it counts most. Kept in one place for tuning.
TITLE_WEIGHT = 3.0     # <title>
HEADING_WEIGHT = 2.0   # <h1>, <h2>, <h3>
BOLD_WEIGHT = 1.5      # <b>, <strong>

# How strongly long pages are penalized. A page's score is divided by
# (vector length ** LENGTH_POWER). 1.0 is full cosine similarity, 0 is no penalty.
# Full cosine pushed useful long pages (the CS home page, the course list) out of the
# top results, so the square root (0.5) is used as a softer penalty.
LENGTH_POWER = 0.5

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
    # Keep the index (term -> postings) and the doc ID -> URL map, and work out every
    # page's length divisor once here so no query has to.
    def __init__(self, index: dict, doc_ids: dict):
        self.index = index
        self.urls = {int(k): v for k, v in doc_ids.items()}
        self.num_docs = len(self.urls)
        self.length_divisor = {
            doc_id: length ** LENGTH_POWER for doc_id, length in _doc_lengths(index).items()
        }

    # Build an engine from the files the indexer saved in output_dir.
    @classmethod
    def from_dir(cls, output_dir) -> "SearchEngine":
        output_dir = Path(output_dir)
        index = json.loads((output_dir / INDEX_FILE).read_text(encoding="utf-8"))
        doc_ids = json.loads((output_dir / DOC_IDS_FILE).read_text(encoding="utf-8"))
        return cls(index, doc_ids)

    # Return the top k pages for a query: more matched terms first, then higher score.
    # Score = sum over query terms of idf * term weight / length divisor.
    def search(self, query: str, k: int = 5) -> list[Result]:
        scores: dict[int, float] = {}
        matched: dict[int, int] = {}
        for term in dict.fromkeys(tokenize(query)):  # unique terms, in order
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
        # Only the best k are needed, so pick them without sorting every match.
        top = heapq.nsmallest(k, scores, key=lambda d: (-matched[d], -scores[d], d))
        return [Result(d, self.urls[d], scores[d]) for d in top]


# Log-damped count: 0 stays 0, 1 -> 1, 10 -> 2, 100 -> 3. The 10th repeat of a word
# adds much less than the first, so a page can't win just by repeating a word.
def _damp(count: int) -> float:
    return 1 + math.log10(count) if count > 0 else 0.0


# Weight of one term in one page (before idf): damped tf plus a bonus for each kind of
# important text the term appears in.
def _term_weight(posting: dict) -> float:
    return (_damp(posting["tf"])
            + TITLE_WEIGHT * _damp(posting["title_tf"])
            + HEADING_WEIGHT * _damp(posting["heading_tf"])
            + BOLD_WEIGHT * _damp(posting["bold_tf"]))


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

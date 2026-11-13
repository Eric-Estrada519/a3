"""Ranked retrieval over the Milestone 1 index."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from indexer import DOC_IDS_FILE, INDEX_FILE
from tokenizer import tokenize

# Extra weight for occurrences in title, h1-h3, b or strong.
IMPORTANT_BOOST = 1.0


# One ranked search hit.
@dataclass
class Result:
    doc_id: int
    url: str
    score: float


# Holds the index in memory and answers queries against it.
class SearchEngine:
    # Keep the index (term -> postings) and the doc ID -> URL map.
    def __init__(self, index: dict, doc_ids: dict):
        self.index = index
        self.urls = {int(k): v for k, v in doc_ids.items()}
        self.num_docs = len(self.urls)

    # Build an engine from the files the indexer saved in output_dir.
    @classmethod
    def from_dir(cls, output_dir) -> "SearchEngine":
        output_dir = Path(output_dir)
        index = json.loads((output_dir / INDEX_FILE).read_text(encoding="utf-8"))
        doc_ids = json.loads((output_dir / DOC_IDS_FILE).read_text(encoding="utf-8"))
        return cls(index, doc_ids)

    # Return the top k pages for a query: more matched terms first, then higher score.
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
                weight = _term_weight(posting["tf"], posting["important_tf"], idf)
                scores[doc_id] = scores.get(doc_id, 0.0) + weight
                matched[doc_id] = matched.get(doc_id, 0) + 1
        ranked = sorted(scores, key=lambda d: (-matched[d], -scores[d], d))
        return [Result(d, self.urls[d], scores[d]) for d in ranked[:k]]


# tf-idf weight of one term in one page, plus a boost for important text.
def _term_weight(tf: int, important_tf: int, idf: float) -> float:
    weight = (1 + math.log10(tf)) * idf
    if important_tf:
        weight += IMPORTANT_BOOST * (1 + math.log10(important_tf)) * idf
    return weight

"""Build the inverted index: python indexer.py [dataset_dir] [output_dir]"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from parser import PageError, defragment, load_page, parse_page

# Output file names (search.py imports these too).
INDEX_FILE = "index.json"
NGRAMS_FILE = "ngrams.json"
DOC_IDS_FILE = "doc_ids.json"
SKIPPED_FILE = "skipped.json"
PAGERANK_FILE = "pagerank.json"

# PageRank settings. DAMPING is the chance that a random surfer follows a link on the
# page instead of jumping to any page at random; 0.85 is the usual value.
DAMPING = 0.85
PAGERANK_MAX_ITERATIONS = 100
PAGERANK_TOLERANCE = 1e-10   # stop once the ranks change by less than this in total


# One posting: (doc_id, tf, title_tf, heading_tf, bold_tf, anchor_tf).
# anchor_tf counts the word in the text of links from OTHER pages that point to this
# page. It is not part of tf, so a word found only in links has tf 0.
Posting = tuple[int, int, int, int, int, int]

# One phrase posting: (doc_id, number of times the 2- or 3-word phrase appears in the page).
NgramPosting = tuple[int, int]


# Holds the in-memory index while it is being built.
class IndexBuilder:
    # Start empty. Counts are kept as term -> {doc_id: [counts]} so anchor text can be
    # added to any page later; postings() turns this into lists sorted by doc_id.
    def __init__(self):
        self._counts: dict[str, dict[int, list[int]]] = {}
        self._ngram_counts: dict[str, dict[int, int]] = {}
        self._last_doc_id = -1

    # Add one page's term counts. Each page is added once, in increasing doc_id order.
    # counts is {term: (tf, title_tf, heading_tf, bold_tf)} from the parser.
    def add_document(self, doc_id: int, counts: dict[str, tuple[int, int, int, int]]) -> None:
        if doc_id <= self._last_doc_id:
            raise ValueError(
                f"doc_id {doc_id} must be greater than previous {self._last_doc_id}"
            )
        self._last_doc_id = doc_id
        for term, term_counts in counts.items():
            self._counts.setdefault(term, {})[doc_id] = [*term_counts, 0]

    # Add the words of one link that points to page doc_id (one anchor_tf per word).
    def add_anchor_text(self, doc_id: int, words: list[str]) -> None:
        for word in words:
            entry = self._counts.setdefault(word, {}).setdefault(doc_id, [0, 0, 0, 0, 0])
            entry[4] += 1

    # Add one page's phrase counts: {"word1 word2": count, "word1 word2 word3": count}.
    def add_ngrams(self, doc_id: int, ngram_counts: dict[str, int]) -> None:
        for pair, count in ngram_counts.items():
            self._ngram_counts.setdefault(pair, {})[doc_id] = count

    # Return the index built so far: term -> postings sorted by doc_id.
    def postings(self) -> dict[str, list[Posting]]:
        return {
            term: [(doc_id, *docs[doc_id]) for doc_id in sorted(docs)]
            for term, docs in self._counts.items()
        }

    # Return the phrase index: phrase -> [(doc_id, tf)] sorted by doc_id.
    def ngram_postings(self) -> dict[str, list[NgramPosting]]:
        return {
            pair: [(doc_id, docs[doc_id]) for doc_id in sorted(docs)]
            for pair, docs in self._ngram_counts.items()
        }


# Convert tuple postings to the saved shape with named fields, terms sorted.
def postings_to_json(postings: dict[str, list[Posting]]) -> dict:
    return {
        term: [
            {"doc_id": doc_id, "tf": tf, "title_tf": title_tf,
             "heading_tf": heading_tf, "bold_tf": bold_tf, "anchor_tf": anchor_tf}
            for doc_id, tf, title_tf, heading_tf, bold_tf, anchor_tf in postings[term]
        ]
        for term in sorted(postings)
    }


# Convert phrase postings to the saved shape {"doc_id", "tf"}, phrases sorted.
def ngram_postings_to_json(ngram_postings: dict[str, list[NgramPosting]]) -> dict:
    return {
        pair: [{"doc_id": doc_id, "tf": tf} for doc_id, tf in ngram_postings[pair]]
        for pair in sorted(ngram_postings)
    }


# Everything produced by one indexing run.
@dataclass
class BuildResult:
    index: dict[str, list[Posting]]
    doc_map: dict[int, str]                            # key: doc_id, value:URL
    skipped: list[dict] = field(default_factory=list)  # [{"file", "reason"}]
    total_files: int = 0
    ngram_index: dict[str, list[NgramPosting]] = field(default_factory=dict)
    pagerank: dict[int, float] = field(default_factory=dict)  # key: doc_id


# PageRank of every page from the links between pages: {doc_id: rank}, summing to 1.
# links holds (source doc_id, target doc_id) pairs, each pair once, no self-links.
# A page with no links out spreads its rank over all pages, so no rank is lost.
def compute_pagerank(num_docs: int, links: set[tuple[int, int]]) -> dict[int, float]:
    if num_docs == 0:
        return {}
    out_links: dict[int, list[int]] = {}
    for source, target in sorted(links):   # sorted so every run adds in the same order
        out_links.setdefault(source, []).append(target)
    no_links_out = [doc_id for doc_id in range(num_docs) if doc_id not in out_links]
    ranks = [1 / num_docs] * num_docs
    for _ in range(PAGERANK_MAX_ITERATIONS):
        spread = sum(ranks[doc_id] for doc_id in no_links_out) / num_docs
        new_ranks = [(1 - DAMPING) / num_docs + DAMPING * spread] * num_docs
        for source, targets in out_links.items():
            share = DAMPING * ranks[source] / len(targets)
            for target in targets:
                new_ranks[target] += share
        change = sum(abs(new - old) for new, old in zip(new_ranks, ranks))
        ranks = new_ranks
        if change < PAGERANK_TOLERANCE:
            break
    return dict(enumerate(ranks))


# Yield every .json file under the dataset folder in sorted (reproducible) order.
def iter_json_files(dataset_dir):
    yield from sorted(Path(dataset_dir).rglob("*.json"))


# Read every page, skip bad or duplicate ones, assign doc IDs, and build the index.
# Anchor text is added at the end, once every page's URL has a doc ID.
def build_index(dataset_dir) -> BuildResult:
    dataset_dir = Path(dataset_dir)
    builder = IndexBuilder()
    doc_map: dict[int, str] = {}
    skipped: list[dict] = []
    first_file_for_url: dict[str, str] = {}
    first_page_for_fingerprint: dict[str, tuple[str, int]] = {}  # hash -> (file, doc_id)
    doc_for_url: dict[str, int] = {}     # every known URL -> the doc ID that holds it
    links: list[tuple[int, str, list[str]]] = []   # (source doc_id, target URL, words)
    total = 0

    for path in iter_json_files(dataset_dir):
        total += 1
        rel = path.relative_to(dataset_dir).as_posix()
        try:
            url, content = load_page(path)
            url = defragment(url)
            # Same URL once the #fragment is removed. Keeps only the first file.
            if url in first_file_for_url:
                raise PageError(f"duplicate URL (first: {first_file_for_url[url]})")
            first_file_for_url[url] = rel
            page = parse_page(content, url)
            if not page.counts:
                raise PageError("no tokens")
            # Exact same words as a page already indexed (e.g. /mars/ and
            # /mars/index.shtml). Links to this URL will count for the kept page.
            if page.fingerprint in first_page_for_fingerprint:
                first_rel, kept_doc_id = first_page_for_fingerprint[page.fingerprint]
                doc_for_url[url] = kept_doc_id
                raise PageError(f"duplicate content (first: {first_rel})")
        except PageError as err:
            skipped.append({"file": rel, "reason": err.reason})
            continue

        doc_id = len(doc_map)
        doc_map[doc_id] = url
        doc_for_url[url] = doc_id
        first_page_for_fingerprint[page.fingerprint] = (rel, doc_id)
        builder.add_document(doc_id, page.counts)
        builder.add_ngrams(doc_id, page.ngram_counts)
        links += [(doc_id, target, words) for target, words in page.links]

    # Credit each link's words to the page it points to. Links to pages outside the
    # dataset, and links from a page to itself, are ignored. The same links (each
    # source -> target pair once) make the link graph for PageRank.
    link_pairs: set[tuple[int, int]] = set()
    for source_doc_id, target_url, words in links:
        target_doc_id = doc_for_url.get(target_url)
        if target_doc_id is not None and target_doc_id != source_doc_id:
            builder.add_anchor_text(target_doc_id, words)
            link_pairs.add((source_doc_id, target_doc_id))

    return BuildResult(builder.postings(), doc_map, skipped, total, builder.ngram_postings(),
                       compute_pagerank(len(doc_map), link_pairs))


# Write UTF-8 JSON. Compact by default to keep the index small.
def _write_json(path: Path, data, compact: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Save the index, the phrase index, the doc ID to URL map, the PageRank of each page,
# and the (readable) skip log.
def write_outputs(result: BuildResult, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / INDEX_FILE, postings_to_json(result.index))
    _write_json(output_dir / NGRAMS_FILE, ngram_postings_to_json(result.ngram_index))
    _write_json(output_dir / DOC_IDS_FILE, {str(k): v for k, v in result.doc_map.items()})
    _write_json(output_dir / PAGERANK_FILE, {str(k): v for k, v in result.pagerank.items()})
    _write_json(output_dir / SKIPPED_FILE, result.skipped, compact=False)


# read the two folder paths, build the index, save it. This runs via the command line
def main(argv=None) -> int:
    arg_parser = argparse.ArgumentParser(description="Build the inverted index.")
    arg_parser.add_argument("dataset_dir", nargs="?", default="analyst/ANALYST",
                            help="folder of JSON pages (default: analyst/ANALYST)")
    arg_parser.add_argument("output_dir", nargs="?", default="output",
                            help="where to save the index (default: output)")
    args = arg_parser.parse_args(argv)
    # A wrong path would otherwise "succeed" with 0 files and save an empty index.
    if not Path(args.dataset_dir).is_dir():
        arg_parser.error(f"dataset folder not found: {args.dataset_dir}")

    result = build_index(args.dataset_dir)
    write_outputs(result, args.output_dir)
    print(f"Indexed {len(result.doc_map)} of {result.total_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

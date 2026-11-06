"""run the inverted index by running python indexer.py dataset_dir output_dir"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from parser import PageError, defragment, extract_term_counts, load_page

# Output file names (report.py imports these too).
INDEX_FILE = "index.json"
DOC_IDS_FILE = "doc_ids.json"
SKIPPED_FILE = "skipped.json"


# Holds the in-memory index so term maps to list of (doc_id, tf, important_tf).
class IndexBuilder:
    # Start with an empty index and no documents seen yet.
    def __init__(self):
        self._postings: dict[str, list[tuple[int, int, int]]] = {}
        self._last_doc_id = -1

    # Add one page's term counts. doc IDs must increase so postings stay sorted.
    def add_document(self, doc_id: int, counts: dict[str, tuple[int, int]]) -> None:
        if doc_id <= self._last_doc_id:
            raise ValueError(
                f"doc_id {doc_id} must be greater than previous {self._last_doc_id}"
            )
        self._last_doc_id = doc_id
        for term, (tf, important_tf) in counts.items():
            self._postings.setdefault(term, []).append((doc_id, tf, important_tf))

    # Return the index built so far.
    def postings(self) -> dict[str, list[tuple[int, int, int]]]:
        return self._postings


# Convert tuple postings to the saved {"doc_id", "tf", "important_tf"} shape, terms sorted.
def postings_to_json(postings: dict[str, list[tuple[int, int, int]]]) -> dict:
    return {
        term: [
            {"doc_id": doc_id, "tf": tf, "important_tf": important_tf}
            for doc_id, tf, important_tf in postings[term]
        ]
        for term in sorted(postings)
    }


# Everything produced by one indexing run.
@dataclass
class BuildResult:
    index: dict[str, list[tuple[int, int, int]]]
    doc_map: dict[int, str]                            # key: doc_id, value:URL
    skipped: list[dict] = field(default_factory=list)  # [{"file", "reason"}]
    total_files: int = 0


# Yield every .json file under the dataset folder in sorted (reproducible) order.
def iter_json_files(dataset_dir):
    yield from sorted(Path(dataset_dir).rglob("*.json"))


# Read every page and skips bad or duplicate ones, assigns doc IDs, and build the index.
def build_index(dataset_dir) -> BuildResult:
    dataset_dir = Path(dataset_dir)
    builder = IndexBuilder()
    doc_map: dict[int, str] = {}
    skipped: list[dict] = []
    first_file_for_url: dict[str, str] = {}
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
            counts = extract_term_counts(content)
            if not counts:
                raise PageError("no tokens")
        except PageError as err:
            skipped.append({"file": rel, "reason": err.reason})
            continue

        doc_id = len(doc_map)
        doc_map[doc_id] = url
        builder.add_document(doc_id, counts)

    return BuildResult(builder.postings(), doc_map, skipped, total)


# Write UTF-8 JSON. Compact by default to keep the index small.
def _write_json(path: Path, data, compact: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Save the index, the doc ID to URL map, and the (readable) skip log.
def write_outputs(result: BuildResult, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / INDEX_FILE, postings_to_json(result.index))
    _write_json(output_dir / DOC_IDS_FILE, {str(k): v for k, v in result.doc_map.items()})
    _write_json(output_dir / SKIPPED_FILE, result.skipped, compact=False)


# read the two folder paths, build the index, save it. This runs via the command line
def main(argv=None) -> int:
    arg_parser = argparse.ArgumentParser(description="Build the inverted index.")
    arg_parser.add_argument("dataset_dir", help="folder of JSON pages")
    arg_parser.add_argument("output_dir", help="where to save the index")
    args = arg_parser.parse_args(argv)

    result = build_index(args.dataset_dir)
    write_outputs(result, args.output_dir)
    print(f"Indexed {len(result.doc_map)} of {result.total_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

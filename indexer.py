import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from parser import PageError, defragment, extract_term_counts, load_page

INDEX_FILE = "index.json"
DOC_IDS_FILE = "doc_ids.json"
SKIPPED_FILE = "skipped.json"


class IndexBuilder:
    def __init__(self):
        self._postings: dict[str, list[tuple[int, int, int]]] = {}
        self._last_doc_id = -1

    def add_document(self, doc_id: int, counts: dict[str, tuple[int, int]]) -> None:
        if doc_id <= self._last_doc_id:
            raise ValueError(
                f"doc_id {doc_id} must be greater than previous {self._last_doc_id}"
            )
        self._last_doc_id = doc_id
        for term, (tf, important_tf) in counts.items():
            self._postings.setdefault(term, []).append((doc_id, tf, important_tf))

    def postings(self) -> dict[str, list[tuple[int, int, int]]]:
        return self._postings


def postings_to_json(postings: dict[str, list[tuple[int, int, int]]]) -> dict:
    return {
        term: [
            {"doc_id": doc_id, "tf": tf, "important_tf": important_tf}
            for doc_id, tf, important_tf in postings[term]
        ]
        for term in sorted(postings)
    }


@dataclass
class BuildResult:
    index: dict[str, list[tuple[int, int, int]]]
    doc_map: dict[int, str]
    skipped: list[dict] = field(default_factory=list)
    total_files: int = 0


def iter_json_files(dataset_dir):
    yield from sorted(Path(dataset_dir).rglob("*.json"))


def build_index(dataset_dir, limit: int | None = None) -> BuildResult:
    dataset_dir = Path(dataset_dir)
    builder = IndexBuilder()
    doc_map: dict[int, str] = {}
    skipped: list[dict] = []
    first_file_for_url: dict[str, str] = {}
    total = 0

    for path in islice(iter_json_files(dataset_dir), limit):
        total += 1
        rel = path.relative_to(dataset_dir).as_posix()
        try:
            url, content = load_page(path)
            url = defragment(url)
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


def _write_json(path: Path, data, compact: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)





def _reason_category(reason: str) -> str:
    return reason.split(" (")[0].split(":")[0]


if __name__ == "__main__":
    raise SystemExit(main())

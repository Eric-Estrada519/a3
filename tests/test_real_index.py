"""Checks the real index in output/ against the dataset (skipped if either is missing)."""
import json
import os
import re
from pathlib import Path

import pytest

from indexer import (NGRAMS_FILE, DOC_IDS_FILE, INDEX_FILE, PAGERANK_FILE, SKIPPED_FILE,
                     iter_json_files)
from parser import defragment, extract_term_counts, load_page, parse_page
from search import SearchEngine

# Folders to check; override with A3_OUTPUT_DIR / A3_DATASET_DIR (e.g. to test a broken copy).
OUTPUT_DIR = Path(os.environ.get("A3_OUTPUT_DIR", "output"))
DATASET_DIR = Path(os.environ.get("A3_DATASET_DIR", "analyst/ANALYST"))

# Every 25th document is re-parsed from its JSON file and compared with the index.
SAMPLE_EVERY = 25

pytestmark = pytest.mark.skipif(
    not (OUTPUT_DIR / INDEX_FILE).is_file() or not DATASET_DIR.is_dir(),
    reason="real index or dataset not found",
)


# Load the saved index once for all tests in this file.
@pytest.fixture(scope="module")
def index():
    return json.loads((OUTPUT_DIR / INDEX_FILE).read_text(encoding="utf-8"))


# Load the saved doc ID -> URL map once.
@pytest.fixture(scope="module")
def doc_ids():
    return json.loads((OUTPUT_DIR / DOC_IDS_FILE).read_text(encoding="utf-8"))


# Load the saved skip log once.
@pytest.fixture(scope="module")
def skipped():
    return json.loads((OUTPUT_DIR / SKIPPED_FILE).read_text(encoding="utf-8"))


# Map each indexed URL to the first dataset file with that URL (the indexer's rule).
@pytest.fixture(scope="module")
def file_for_url():
    first = {}
    for path in iter_json_files(DATASET_DIR):
        try:
            url, _ = load_page(path)
        except Exception:
            continue
        first.setdefault(defragment(url), path)
    return first


# Load the saved word-pair index once.
@pytest.fixture(scope="module")
def ngram_index():
    return json.loads((OUTPUT_DIR / NGRAMS_FILE).read_text(encoding="utf-8"))


# Turn the inverted index around: doc_id -> {term: (tf, title_tf, heading_tf, bold_tf)}.
# Words a page only gets from links pointing to it (tf 0) are left out, because they
# are not in the page's own HTML.
@pytest.fixture(scope="module")
def forward(index):
    docs = {}
    for term, postings in index.items():
        for p in postings:
            if p["tf"] > 0:
                docs.setdefault(p["doc_id"], {})[term] = (
                    p["tf"], p["title_tf"], p["heading_tf"], p["bold_tf"])
    return docs


def test_every_file_is_indexed_or_skipped(doc_ids, skipped):
    total = sum(1 for _ in iter_json_files(DATASET_DIR))
    assert len(doc_ids) + len(skipped) == total


def test_doc_ids_are_0_to_n_minus_1(doc_ids):
    assert sorted(int(k) for k in doc_ids) == list(range(len(doc_ids)))


def test_urls_are_unique_and_have_no_fragment(doc_ids):
    urls = list(doc_ids.values())
    assert len(set(urls)) == len(urls)
    assert not [u for u in urls if "#" in u]


def test_terms_are_lowercase_alphanumeric(index):
    bad = [t for t in index if not re.fullmatch(r"[a-z0-9]+", t)]
    assert bad == []


def test_postings_are_sorted_unique_and_point_to_real_docs(index, doc_ids):
    problems = []
    for term, postings in index.items():
        ids = [p["doc_id"] for p in postings]
        if ids != sorted(set(ids)):
            problems.append((term, "not sorted/unique"))
        if any(str(i) not in doc_ids for i in ids):
            problems.append((term, "unknown doc_id"))
    assert problems == []


def test_counts_are_valid(index):
    bad = [
        (term, p) for term, postings in index.items() for p in postings
        if set(p) != {"doc_id", "tf", "title_tf", "heading_tf", "bold_tf", "anchor_tf"}
        or p["tf"] + p["anchor_tf"] < 1
        or min(p["tf"], p["title_tf"], p["heading_tf"], p["bold_tf"], p["anchor_tf"]) < 0
        or p["title_tf"] + p["heading_tf"] + p["bold_tf"] > p["tf"]
    ]
    assert bad == []


def test_every_document_has_postings(forward, doc_ids):
    assert set(forward) == {int(k) for k in doc_ids}


def test_sampled_documents_match_a_fresh_parse(forward, doc_ids, file_for_url):
    # Re-parse every SAMPLE_EVERY-th page from its JSON file and compare all counts.
    mismatches = []
    for doc_id in range(0, len(doc_ids), SAMPLE_EVERY):
        url = doc_ids[str(doc_id)]
        _, content = load_page(file_for_url[url])
        if extract_term_counts(content) != forward[doc_id]:
            mismatches.append((doc_id, url))
    assert mismatches == []


def test_skipped_duplicates_really_share_an_indexed_url(skipped, doc_ids):
    # The shared URL is indexed, unless its first file was itself skipped as a copy of
    # another page's content (then that content is indexed under the other URL).
    indexed = set(doc_ids.values())
    reason_for_file = {entry["file"]: entry["reason"] for entry in skipped}
    for entry in skipped:
        if entry["reason"].startswith("duplicate URL"):
            url, _ = load_page(DATASET_DIR / entry["file"])
            first = entry["reason"].split("(first: ")[1].rstrip(")")
            assert (defragment(url) in indexed
                    or reason_for_file.get(first, "").startswith("duplicate content")), entry["file"]


def test_skipped_content_duplicates_have_the_same_words_as_the_kept_page(skipped):
    checked = 0
    for entry in skipped:
        if entry["reason"].startswith("duplicate content"):
            first = entry["reason"].split("(first: ")[1].rstrip(")")
            copy_url, copy_html = load_page(DATASET_DIR / entry["file"])
            kept_url, kept_html = load_page(DATASET_DIR / first)
            assert (parse_page(copy_html, copy_url).fingerprint
                    == parse_page(kept_html, kept_url).fingerprint), entry["file"]
            checked += 1
    assert checked > 0   # the dataset does have exact duplicates


def test_indexed_pages_have_no_exact_duplicates(doc_ids, file_for_url):
    seen = {}
    for doc_id in range(0, len(doc_ids), SAMPLE_EVERY):
        url = doc_ids[str(doc_id)]
        _, content = load_page(file_for_url[url])
        fingerprint = parse_page(content, url).fingerprint
        assert fingerprint not in seen, (url, seen.get(fingerprint))
        seen[fingerprint] = url


def test_word_pairs_point_to_real_docs_that_contain_both_words(ngram_index, forward, doc_ids):
    assert ngram_index["ic 31"]                 # the pair from the "ics 31" query exists
    problems = []
    for pair in ("ic 31", "machin learn", "softwar engin"):
        first, second = pair.split()
        for p in ngram_index[pair]:
            words = forward.get(p["doc_id"], {})
            if str(p["doc_id"]) not in doc_ids or first not in words or second not in words:
                problems.append((pair, p))
    assert problems == []


def test_some_pages_get_words_from_links(index):
    anchor_only = sum(1 for postings in index.values() for p in postings
                      if p["tf"] == 0 and p["anchor_tf"] > 0)
    assert anchor_only > 0


def test_phd_program_finds_a_phd_program_page():
    top = [r.url for r in SearchEngine.from_dir(OUTPUT_DIR).search("phd program")]
    assert any("/grad/phd-" in url for url in top[:3]), top


def test_ics_31_finds_the_course_list_first():
    top = [r.url for r in SearchEngine.from_dir(OUTPUT_DIR).search("ics 31")]
    assert top[0].endswith("/undergrad/courses/"), top


def test_how_to_apply_finds_an_admissions_page():
    top = [r.url for r in SearchEngine.from_dir(OUTPUT_DIR).search("how to apply")]
    assert any("admissions" in url or "apply" in url for url in top), top


def test_known_queries_find_expected_pages():
    engine = SearchEngine.from_dir(OUTPUT_DIR)
    top = {q: [r.url for r in engine.search(q)] for q in
           ("cristina lopes", "master of software engineering")}
    assert top["cristina lopes"][0].endswith("/faculty-profiles/cristina-lopes/")
    assert top["master of software engineering"][0].endswith("/grad/mswe/")


def test_every_page_has_a_pagerank_and_they_sum_to_one(doc_ids):
    pagerank = json.loads((OUTPUT_DIR / PAGERANK_FILE).read_text(encoding="utf-8"))
    assert set(pagerank) == set(doc_ids)
    assert sum(pagerank.values()) == pytest.approx(1.0)
    assert min(pagerank.values()) > 0


def test_pagerank_lifts_the_menu_pages_people_look_for():
    engine = SearchEngine.from_dir(OUTPUT_DIR)
    assert engine.link_boost                    # the saved PageRank file was loaded
    top = {q: [r.url for r in engine.search(q)] for q in ("about", "faculty", "undergraduate")}
    assert top["about"][0].endswith("cs.uci.edu/about-the-department/"), top["about"]
    assert "https://www.cs.uci.edu/faculty/" in top["faculty"][:3], top["faculty"]
    assert "https://www.cs.uci.edu/undergraduate-programs/" in top["undergraduate"][:3]

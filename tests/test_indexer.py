"""Tests for indexer.py: posting construction, dataset walk, outputs, CLI."""
import json

import pytest

from indexer import (
    NGRAMS_FILE,
    DOC_IDS_FILE,
    INDEX_FILE,
    PAGERANK_FILE,
    SKIPPED_FILE,
    IndexBuilder,
    build_index,
    compute_pagerank,
    iter_json_files,
    main,
    postings_to_json,
)


def test_builder_collects_postings_per_term():
    builder = IndexBuilder()
    builder.add_document(0, {"cat": (2, 1, 0, 0), "dog": (1, 0, 0, 1)})
    builder.add_document(1, {"cat": (1, 0, 0, 0)})
    assert builder.postings() == {
        "cat": [(0, 2, 1, 0, 0, 0), (1, 1, 0, 0, 0, 0)],
        "dog": [(0, 1, 0, 0, 1, 0)],
    }


def test_builder_rejects_doc_ids_out_of_order():
    # Postings are only guaranteed sorted if doc IDs increase.
    builder = IndexBuilder()
    builder.add_document(5, {"cat": (1, 0, 0, 0)})
    with pytest.raises(ValueError):
        builder.add_document(5, {"cat": (1, 0, 0, 0)})
    with pytest.raises(ValueError):
        builder.add_document(3, {"cat": (1, 0, 0, 0)})


def test_postings_to_json_sorts_terms_and_names_fields():
    postings = {"zebra": [(1, 1, 0, 0, 0, 0)],
                "apple": [(0, 3, 1, 1, 0, 0), (1, 0, 0, 0, 0, 2)]}
    result = postings_to_json(postings)
    assert list(result) == ["apple", "zebra"]
    assert result["apple"] == [
        {"doc_id": 0, "tf": 3, "title_tf": 1, "heading_tf": 1, "bold_tf": 0, "anchor_tf": 0},
        {"doc_id": 1, "tf": 0, "title_tf": 0, "heading_tf": 0, "bold_tf": 0, "anchor_tf": 2},
    ]


def _page(url, content, encoding="utf-8"):
    return json.dumps({"url": url, "content": content, "encoding": encoding})


@pytest.fixture
def dataset(tmp_path):
    """Six files in two domain folders covering every skip reason.

    Sorted order (and therefore doc-ID order):
      a_domain/01.json  indexed   -> doc 0
      a_domain/02.json  duplicate of 01 (URL differs only by fragment)
      a_domain/03.json  invalid JSON
      b_domain/01.json  indexed   -> doc 1
      b_domain/02.json  missing content
      b_domain/03.json  no tokens (script only)
    """
    root = tmp_path / "data"
    a, b = root / "a_domain", root / "b_domain"
    a.mkdir(parents=True)
    b.mkdir()
    (a / "01.json").write_text(_page(
        "https://a.com/page#top",
        "<html><head><title>Hello</title></head><body><p>hello world</p></body></html>",
    ), encoding="utf-8")
    (a / "02.json").write_text(_page("https://a.com/page#bottom", "<p>other</p>"),
                               encoding="utf-8")
    (a / "03.json").write_text("{not json", encoding="utf-8")
    (b / "01.json").write_text(_page("https://b.com/", "<h1>World</h1><p>running</p>"),
                               encoding="utf-8")
    (b / "02.json").write_text(json.dumps({"url": "https://b.com/x", "encoding": "utf-8"}),
                               encoding="utf-8")
    (b / "03.json").write_text(_page("https://b.com/empty", "<script>var x;</script>"),
                               encoding="utf-8")
    (root / "notes.txt").write_text("not a page", encoding="utf-8")  # ignored: not .json
    return root


def test_iter_json_files_is_sorted_and_only_json(dataset):
    names = [p.relative_to(dataset).as_posix() for p in iter_json_files(dataset)]
    assert names == ["a_domain/01.json", "a_domain/02.json", "a_domain/03.json",
                     "b_domain/01.json", "b_domain/02.json", "b_domain/03.json"]


def test_build_index_assigns_doc_ids_and_defragments_urls(dataset):
    result = build_index(dataset)
    assert result.doc_map == {0: "https://a.com/page", 1: "https://b.com/"}


def test_build_index_counts_terms(dataset):
    index = build_index(dataset).index
    assert index["hello"] == [(0, 2, 1, 0, 0, 0)]                       # title + body
    assert index["world"] == [(0, 1, 0, 0, 0, 0), (1, 1, 0, 1, 0, 0)]   # body on 0, h1 on 1
    assert index["run"] == [(1, 1, 0, 0, 0, 0)]                         # "running" is stemmed
    assert "other" not in index                          # duplicate page not indexed
    assert "var" not in index                            # script text removed


def test_build_index_logs_every_skip_with_reason(dataset):
    skipped = {s["file"]: s["reason"] for s in build_index(dataset).skipped}
    assert skipped == {
        "a_domain/02.json": "duplicate URL (first: a_domain/01.json)",
        "a_domain/03.json": "invalid JSON",
        "b_domain/02.json": "missing content",
        "b_domain/03.json": "no tokens",
    }


def test_indexed_plus_skipped_equals_total(dataset):
    result = build_index(dataset)
    assert result.total_files == 6
    assert len(result.doc_map) + len(result.skipped) == result.total_files


def test_postings_are_sorted_by_doc_id(dataset):
    for postings in build_index(dataset).index.values():
        doc_ids = [posting[0] for posting in postings]
        assert doc_ids == sorted(doc_ids)


def test_important_counts_never_exceed_tf_in_index(dataset):
    for postings in build_index(dataset).index.values():
        for _, tf, title_tf, heading_tf, bold_tf, _ in postings:
            assert title_tf + heading_tf + bold_tf <= tf


def test_main_writes_five_output_files(dataset, tmp_path):
    out = tmp_path / "out"
    assert main([str(dataset), str(out)]) == 0
    for name in (INDEX_FILE, NGRAMS_FILE, DOC_IDS_FILE, SKIPPED_FILE, PAGERANK_FILE):
        assert (out / name).is_file()


def test_index_reloads_from_disk_with_named_fields(dataset, tmp_path):
    out = tmp_path / "out"
    main([str(dataset), str(out)])
    index = json.loads((out / INDEX_FILE).read_text(encoding="utf-8"))
    assert index["world"] == [
        {"doc_id": 0, "tf": 1, "title_tf": 0, "heading_tf": 0, "bold_tf": 0, "anchor_tf": 0},
        {"doc_id": 1, "tf": 1, "title_tf": 0, "heading_tf": 1, "bold_tf": 0, "anchor_tf": 0},
    ]


def test_every_posting_doc_id_maps_to_a_url(dataset, tmp_path):
    out = tmp_path / "out"
    main([str(dataset), str(out)])
    index = json.loads((out / INDEX_FILE).read_text(encoding="utf-8"))
    doc_ids = json.loads((out / DOC_IDS_FILE).read_text(encoding="utf-8"))
    assert doc_ids == {"0": "https://a.com/page", "1": "https://b.com/"}
    for postings in index.values():
        for posting in postings:
            assert str(posting["doc_id"]) in doc_ids


def test_index_and_doc_ids_are_compact_json(dataset, tmp_path):
    out = tmp_path / "out"
    main([str(dataset), str(out)])
    for name in (INDEX_FILE, NGRAMS_FILE, DOC_IDS_FILE):
        text = (out / name).read_text(encoding="utf-8")
        assert ", " not in text and ": " not in text and "\n" not in text


def test_skipped_file_lists_reasons(dataset, tmp_path):
    out = tmp_path / "out"
    main([str(dataset), str(out)])
    skipped = json.loads((out / SKIPPED_FILE).read_text(encoding="utf-8"))
    assert len(skipped) == 4
    assert {"file": "a_domain/03.json", "reason": "invalid JSON"} in skipped


def test_main_prints_summary(dataset, tmp_path, capsys):
    main([str(dataset), str(tmp_path / "out")])
    printed = capsys.readouterr().out
    assert "Indexed 2 of 6 files" in printed


def test_main_stops_on_missing_dataset_folder(tmp_path):
    # A wrong path must not silently save an empty index.
    with pytest.raises(SystemExit):
        main([str(tmp_path / "no_such_folder"), str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_first_file_claims_url_even_if_it_has_no_tokens(tmp_path):
    # "First file wins": a later copy of the same URL is a duplicate even when the
    # first copy turned out to be empty.
    root = tmp_path / "data" / "d"
    root.mkdir(parents=True)
    (root / "01.json").write_text(_page("https://a.com/#x", "<script>x</script>"),
                                  encoding="utf-8")
    (root / "02.json").write_text(_page("https://a.com/#y", "<p>words</p>"),
                                  encoding="utf-8")
    reasons = [s["reason"] for s in build_index(tmp_path / "data").skipped]
    assert reasons == ["no tokens", "duplicate URL (first: d/01.json)"]


# ----- anchor text, word pairs, and duplicate content -----

def test_builder_adds_anchor_words_to_an_existing_posting():
    builder = IndexBuilder()
    builder.add_document(0, {"cat": (1, 0, 0, 0)})
    builder.add_anchor_text(0, ["cat", "cat"])
    assert builder.postings() == {"cat": [(0, 1, 0, 0, 0, 2)]}


def test_builder_anchor_only_word_gets_tf_zero_and_stays_sorted():
    builder = IndexBuilder()
    builder.add_document(0, {"dog": (1, 0, 0, 0)})
    builder.add_document(1, {"dog": (1, 0, 0, 0)})
    builder.add_anchor_text(1, ["cat"])
    builder.add_anchor_text(0, ["cat"])
    assert builder.postings()["cat"] == [(0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 1)]


def test_builder_collects_ngram_postings():
    builder = IndexBuilder()
    builder.add_ngrams(0, {"ic 31": 2})
    builder.add_ngrams(1, {"ic 31": 1, "31 cours": 1})
    assert builder.ngram_postings() == {"ic 31": [(0, 2), (1, 1)], "31 cours": [(1, 1)]}


def _write_pages(root, pages):
    """pages: list of (relative file name, url, html). Returns the dataset root."""
    for name, url, html in pages:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_page(url, html), encoding="utf-8")
    return root


def test_page_with_same_words_as_earlier_page_is_skipped(tmp_path):
    root = _write_pages(tmp_path / "data", [
        ("d/01.json", "https://a.com/mars/", "<p>multimedia retrieval</p>"),
        ("d/02.json", "https://a.com/mars/index.html", "<div>Multimedia Retrieval</div>"),
        ("d/03.json", "https://a.com/other", "<p>multimedia search</p>"),
    ])
    result = build_index(root)
    assert result.doc_map == {0: "https://a.com/mars/", 1: "https://a.com/other"}
    assert result.skipped == [{"file": "d/02.json", "reason": "duplicate content (first: d/01.json)"}]


def test_anchor_text_is_credited_to_the_linked_page(tmp_path):
    root = _write_pages(tmp_path / "data", [
        ("d/01.json", "https://a.com/", '<p>home</p><a href="/apply">Apply Now</a>'),
        ("d/02.json", "https://a.com/apply", "<h1>Admissions</h1>"),
    ])
    index = build_index(root).index
    # Page 0 shows the link text, so it counts in page 0's own tf as usual.
    # Page 1 gets it as anchor_tf only (tf 0, the word is not on page 1 itself).
    assert index["appli"] == [(0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 1)]
    assert index["now"] == [(0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 1)]


def test_links_to_unknown_pages_and_to_itself_add_nothing(tmp_path):
    root = _write_pages(tmp_path / "data", [
        ("d/01.json", "https://a.com/", '<a href="/">Self</a><a href="https://x.com/">Elsewhere</a>'),
    ])
    index = build_index(root).index
    assert index["self"] == [(0, 1, 0, 0, 0, 0)]
    assert index["elsewher"] == [(0, 1, 0, 0, 0, 0)]


def test_link_to_a_duplicate_copy_credits_the_kept_page(tmp_path):
    root = _write_pages(tmp_path / "data", [
        ("d/01.json", "https://a.com/mars/", "<p>multimedia retrieval</p>"),
        ("d/02.json", "https://a.com/mars/index.html", "<p>multimedia retrieval</p>"),
        ("d/03.json", "https://a.com/", '<a href="/mars/index.html">MARS project</a>'),
    ])
    index = build_index(root).index
    # doc 0 is the kept /mars/ page; doc 1 is the linking page, which shows the text.
    assert index["mar"] == [(0, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 0)]


def test_build_index_collects_word_pairs(tmp_path):
    root = _write_pages(tmp_path / "data", [
        ("d/01.json", "https://a.com/", "<p>ICS 31 and ICS 32</p>"),
        ("d/02.json", "https://a.com/b", "<p>ICS 31</p>"),
    ])
    ngram_index = build_index(root).ngram_index
    assert ngram_index["ic 31"] == [(0, 1), (1, 1)]
    assert ngram_index["ic 32"] == [(0, 1)]
    assert ngram_index["31 and"] == [(0, 1)]
    assert ngram_index["31 and ic"] == [(0, 1)]      # 3-grams are in the same index


def test_ngrams_file_has_named_fields(tmp_path):
    root = _write_pages(tmp_path / "data", [("d/01.json", "https://a.com/", "<p>ICS 31</p>")])
    out = tmp_path / "out"
    main([str(root), str(out)])
    ngram_index = json.loads((out / NGRAMS_FILE).read_text(encoding="utf-8"))
    assert ngram_index == {"ic 31": [{"doc_id": 0, "tf": 1}]}


# ----- PageRank -----

def test_pagerank_of_a_cycle_is_equal_and_sums_to_one():
    ranks = compute_pagerank(3, {(0, 1), (1, 2), (2, 0)})
    assert ranks == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})


def test_pagerank_page_linked_from_two_pages_ranks_highest():
    ranks = compute_pagerank(3, {(0, 2), (1, 2)})
    assert ranks[2] > ranks[0]
    assert ranks[0] == pytest.approx(ranks[1])
    assert sum(ranks.values()) == pytest.approx(1.0)


def test_pagerank_page_with_no_links_out_shares_its_rank():
    # Page 1 links nowhere, so its rank is spread over both pages. Solving by hand:
    # r0 = 0.15/2 + 0.85 * r1/2 and r0 + r1 = 1, so r0 = 0.5 / 1.425.
    ranks = compute_pagerank(2, {(0, 1)})
    assert ranks[0] == pytest.approx(0.5 / 1.425)
    assert ranks[1] == pytest.approx(1 - 0.5 / 1.425)


def test_pagerank_of_empty_dataset_is_empty():
    assert compute_pagerank(0, set()) == {}


def test_build_index_ranks_the_most_linked_page_highest(tmp_path):
    # Both other pages link to the home page; repeated links and a link to itself
    # count once / not at all, so the home page wins only on its incoming links.
    root = _write_pages(tmp_path / "data", [
        ("d/01.json", "https://a.com/", '<p>home</p><a href="/">Home</a><a href="/a">Page A</a>'),
        ("d/02.json", "https://a.com/a", '<p>page a</p><a href="/">Home</a><a href="/">Home</a>'),
        ("d/03.json", "https://a.com/b", '<p>page b</p><a href="/">Home</a>'),
    ])
    ranks = build_index(root).pagerank
    assert set(ranks) == {0, 1, 2}
    assert ranks[0] > ranks[1] > ranks[2]
    assert ranks == pytest.approx(compute_pagerank(3, {(0, 1), (1, 0), (2, 0)}))


def test_main_saves_pagerank_by_doc_id(tmp_path):
    root = _write_pages(tmp_path / "data", [
        ("d/01.json", "https://a.com/", '<p>home</p>'),
        ("d/02.json", "https://a.com/a", '<p>page a</p><a href="/">Home</a>'),
    ])
    out = tmp_path / "out"
    main([str(root), str(out)])
    saved = json.loads((out / PAGERANK_FILE).read_text(encoding="utf-8"))
    assert saved == pytest.approx({"0": 1 - 0.5 / 1.425, "1": 0.5 / 1.425})

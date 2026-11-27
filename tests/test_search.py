"""Tests for search.py: weighted tf-idf with length normalization, and the console."""
import json
import math

import pytest

import search
from indexer import PAGERANK_FILE
from search import SearchEngine, run_console


def urls(results):
    return [r.url for r in results]


def _p(doc_id, tf, title_tf=0, heading_tf=0, bold_tf=0, anchor_tf=0):
    return {"doc_id": doc_id, "tf": tf, "title_tf": title_tf,
            "heading_tf": heading_tf, "bold_tf": bold_tf, "anchor_tf": anchor_tf}


def test_unknown_term_returns_nothing(engine):
    assert engine.search("zebra") == []


def test_empty_or_punctuation_query_returns_nothing(engine):
    assert engine.search("") == []
    assert engine.search("?!--") == []


def test_term_weight_adds_a_bonus_per_kind_of_important_text():
    assert search._term_weight(_p(0, 10)) == pytest.approx(2.0)
    assert search._term_weight(_p(0, 1, title_tf=1)) == pytest.approx(1 + search.TITLE_WEIGHT)
    assert search._term_weight(_p(0, 1, heading_tf=1)) == pytest.approx(1 + search.HEADING_WEIGHT)
    assert search._term_weight(_p(0, 1, bold_tf=1)) == pytest.approx(1 + search.BOLD_WEIGHT)
    assert search._term_weight(_p(0, 0, anchor_tf=1)) == pytest.approx(search.ANCHOR_WEIGHT)


def test_doc_lengths_are_vector_lengths(engine):
    # u0 has cat (tf 3) and the (tf 1): length = sqrt((1 + log10 3)^2 + 1^2).
    lengths = search._doc_lengths(engine.index)
    assert lengths[0] == pytest.approx(math.sqrt((1 + math.log10(3)) ** 2 + 1))
    assert lengths[3] == pytest.approx(1.0)


def test_single_term_score_is_idf_times_weight_over_length(engine):
    idf = math.log10(4 / 2)
    results = engine.search("cat")
    assert urls(results) == ["u0", "u1"]
    expected = idf * (1 + math.log10(3)) / engine.length_divisor[0]
    assert results[0].score == pytest.approx(expected)
    assert results[0].doc_id == 0


def test_query_goes_through_same_tokenizer(engine):
    # "Cats!" -> lowercase, punctuation dropped, stemmed to "cat".
    assert urls(engine.search("Cats!")) == urls(engine.search("cat"))


def test_title_text_boosts_score(engine):
    # Both pages have tf=1 for "dog", but u2's occurrence is in the title.
    assert urls(engine.search("dog")) == ["u2", "u1"]


def test_title_beats_heading_beats_bold_beats_body():
    index = {"word": [_p(0, 1), _p(1, 1, bold_tf=1), _p(2, 1, heading_tf=1),
                      _p(3, 1, title_tf=1)],
             "other": [_p(4, 1)]}
    engine = SearchEngine(index, {str(i): f"u{i}" for i in range(5)})
    assert urls(engine.search("word")) == ["u3", "u2", "u1", "u0"]


def test_long_page_ranks_below_short_page_with_same_match():
    # Both pages contain "cat" once, but page 1 also has 50 other words.
    index = {"cat": [_p(0, 1), _p(1, 1)], "dog": [_p(2, 1)]}
    for i in range(50):
        index[f"filler{i}"] = [_p(1, 1)]
    engine = SearchEngine(index, {"0": "short", "1": "long", "2": "other"})
    assert urls(engine.search("cat")) == ["short", "long"]


def test_pages_with_all_terms_rank_first(engine):
    # u1 has both terms. u2 (title match) only has dog; u0 only has cat.
    assert urls(engine.search("cat dog"))[0] == "u1"


def test_term_in_every_document_has_zero_weight(engine):
    results = engine.search("the")
    assert [r.score for r in results] == [0.0, 0.0, 0.0, 0.0]
    assert urls(results) == ["u0", "u1", "u2", "u3"]   # ties broken by doc_id


def test_k_limits_number_of_results(engine):
    assert urls(engine.search("the", k=2)) == ["u0", "u1"]


def test_repeated_query_terms_count_once(engine):
    assert [r.score for r in engine.search("cat cat")] == [
        r.score for r in engine.search("cat")
    ]


def test_from_dir_loads_saved_index(index_dir):
    engine = SearchEngine.from_dir(index_dir)
    assert engine.num_docs == 4
    assert urls(engine.search("cat")) == ["u0", "u1"]


def test_console_prints_results_skips_blank_lines_and_quits(engine, monkeypatch, capsys):
    answers = iter(["", "dog", "zebra", "quit", "never reached"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    run_console(engine)
    out = capsys.readouterr().out
    assert "1. u2" in out and "2. u1" in out
    assert out.index("1. u2") < out.index("2. u1")
    assert "No results" in out
    assert " ms" in out


def test_console_exits_cleanly_at_end_of_input(engine, monkeypatch):
    def end_of_input(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", end_of_input)
    run_console(engine)   # returns instead of crashing


def test_main_loads_index_then_runs_console(index_dir, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
    assert search.main([str(index_dir)]) == 0
    assert "Loaded 4 pages" in capsys.readouterr().out


# ----- anchor text and word pairs -----

def _engine(index, num_docs, ngram_index=None):
    return SearchEngine(index, {str(i): f"u{i}" for i in range(num_docs)}, ngram_index)


def test_page_found_only_through_link_text():
    # u1 never says "apply" itself, but links to it say "Apply Now".
    index = {"appli": [_p(1, 0, anchor_tf=3)], "other": [_p(0, 1), _p(1, 1)]}
    assert urls(_engine(index, 2).search("apply")) == ["u1"]


def test_link_text_ranks_linked_page_above_page_that_mentions_word_once():
    index = {"faculti": [_p(0, 1), _p(1, 1, anchor_tf=50)], "x": [_p(2, 1)]}
    assert urls(_engine(index, 3).search("faculty")) == ["u1", "u0"]


def _ics31_index():
    # u0 says "ICS ... 31" far apart (more often); u1 says "ICS 31" together.
    # Pages u2-u9 are filler so the pair is rare (1 of 10 pages), as in the real data.
    index = {"ic": [_p(0, 3), _p(1, 1)], "31": [_p(0, 3), _p(1, 1)],
             "x": [_p(i, 1) for i in range(2, 10)]}
    return index, {"ic 31": [{"doc_id": 1, "tf": 1}]}


def test_page_with_the_word_pair_ranks_first():
    index, pairs = _ics31_index()
    assert urls(_engine(index, 10).search("ics 31")) == ["u0", "u1"]          # no pair index
    assert urls(_engine(index, 10, pairs).search("ics 31")) == ["u1", "u0"]   # with it


def test_pair_bonus_needs_the_words_in_query_order():
    index, pairs = _ics31_index()
    assert urls(_engine(index, 10, pairs).search("31 ics")) == ["u0", "u1"]


def test_single_word_query_gets_no_pair_bonus():
    index, pairs = _ics31_index()
    with_pairs = _engine(index, 10, pairs).search("ics")
    without = _engine(index, 10).search("ics")
    assert [(r.url, r.score) for r in with_pairs] == [(r.url, r.score) for r in without]


def test_from_dir_loads_the_pair_index(index_dir):
    engine = SearchEngine.from_dir(index_dir)
    assert engine.ngram_index == {"cat dog": [{"doc_id": 1, "tf": 1}]}


def test_pair_on_too_many_pages_gives_no_bonus():
    # "graduat admiss" is a site-menu phrase on 2 of the 4 pages (more than a quarter),
    # so being side by side says nothing: no bonus, same scores as with no pair index.
    index = {"graduat": [_p(i, 1) for i in range(4)], "admiss": [_p(i, 1) for i in range(4)],
             "x": [_p(4, 1)]}
    pairs = {"graduat admiss": [{"doc_id": 0, "tf": 1}, {"doc_id": 1, "tf": 1}]}
    with_pairs = _engine(index, 5, pairs).search("graduate admissions")
    without = _engine(index, 5).search("graduate admissions")
    assert [(r.url, r.score) for r in with_pairs] == [(r.url, r.score) for r in without]


def test_page_with_the_whole_3_word_phrase_ranks_first():
    # Both pages have "how to" and "to apply" side by side, but only u1 says
    # "how to apply" as one phrase (u0 says "how to ... to apply"). u2-u9 are filler.
    index = {"how": [_p(0, 1), _p(1, 1)], "to": [_p(0, 2), _p(1, 1)],
             "appli": [_p(0, 1), _p(1, 1)], "x": [_p(i, 1) for i in range(2, 10)]}
    pairs = {"how to": [{"doc_id": 0, "tf": 1}, {"doc_id": 1, "tf": 1}],
             "to appli": [{"doc_id": 0, "tf": 1}, {"doc_id": 1, "tf": 1}]}
    with_3gram = dict(pairs, **{"how to appli": [{"doc_id": 1, "tf": 1}]})
    assert urls(_engine(index, 10, pairs).search("how to apply")) == ["u0", "u1"]
    assert urls(_engine(index, 10, with_3gram).search("how to apply")) == ["u1", "u0"]


# ----- PageRank -----

def _tied_index():
    # u0 and u1 match "cat" exactly the same way; u2 is filler.
    return {"cat": [_p(0, 1), _p(1, 1)], "dog": [_p(2, 1)]}


def test_without_pagerank_there_is_no_boost():
    assert search._link_boosts({}, 3) == {}
    assert _engine(_tied_index(), 3).link_boost == {}


def test_top_pagerank_page_gets_the_full_boost():
    boosts = search._link_boosts({"0": 0.7, "1": 0.2, "2": 0.1}, 3)
    assert boosts[0] == pytest.approx(1 + search.PAGERANK_WEIGHT)
    assert 1 < boosts[2] < boosts[1] < boosts[0]


def test_pagerank_breaks_a_tie_between_equal_matches():
    assert urls(_engine(_tied_index(), 3).search("cat")) == ["u0", "u1"]   # by doc ID
    engine = SearchEngine(_tied_index(), {str(i): f"u{i}" for i in range(3)}, None,
                          {"0": 0.1, "1": 0.8, "2": 0.1})
    assert urls(engine.search("cat")) == ["u1", "u0"]


def test_pagerank_multiplies_the_score():
    pagerank = {"0": 0.1, "1": 0.8, "2": 0.1}
    plain = _engine(_tied_index(), 3).search("cat")
    boosted = SearchEngine(_tied_index(), {str(i): f"u{i}" for i in range(3)}, None,
                           pagerank).search("cat")
    boosts = search._link_boosts(pagerank, 3)
    assert boosted[0].score == pytest.approx(plain[0].score * boosts[1])


def test_pagerank_cannot_lift_a_much_weaker_match():
    # u0 has "cat" in its title; u1 only in the body but has almost all the PageRank.
    index = {"cat": [_p(0, 1, title_tf=1), _p(1, 1)], "dog": [_p(2, 1)]}
    engine = SearchEngine(index, {str(i): f"u{i}" for i in range(3)}, None,
                          {"0": 0.01, "1": 0.98, "2": 0.01})
    assert urls(engine.search("cat")) == ["u0", "u1"]


def test_from_dir_loads_pagerank_when_saved(index_dir):
    (index_dir / PAGERANK_FILE).write_text(
        json.dumps({"0": 0.1, "1": 0.7, "2": 0.1, "3": 0.1}), encoding="utf-8")
    engine = SearchEngine.from_dir(index_dir)
    assert engine.link_boost[1] == pytest.approx(1 + search.PAGERANK_WEIGHT)


def test_from_dir_without_pagerank_file_gives_no_boost(index_dir):
    assert SearchEngine.from_dir(index_dir).link_boost == {}

"""Tests for parser.py: URL handling, JSON loading, HTML term counting."""
import json
import warnings

import pytest

from parser import PageError, defragment, extract_term_counts, load_page, parse_page


def test_defragment_removes_fragment():
    assert defragment("https://www.ics.uci.edu/page.html#section-2") == (
        "https://www.ics.uci.edu/page.html"
    )


def test_defragment_leaves_url_without_fragment_unchanged():
    url = "https://www.ics.uci.edu/page.html?x=1"
    assert defragment(url) == url


def test_page_error_keeps_reason():
    err = PageError("invalid JSON")
    assert err.reason == "invalid JSON"
    assert str(err) == "invalid JSON"



def _write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_load_page_returns_url_and_content(tmp_path):
    f = _write_json(tmp_path / "p.json",
                    {"url": "https://a.com/", "content": "<p>hi</p>", "encoding": "utf-8"})
    assert load_page(f) == ("https://a.com/", "<p>hi</p>")


def test_load_page_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(PageError, match="invalid JSON"):
        load_page(f)


def test_load_page_json_that_is_not_an_object(tmp_path):
    f = _write_json(tmp_path / "list.json", ["not", "a", "page"])
    with pytest.raises(PageError, match="invalid JSON"):
        load_page(f)


def test_load_page_missing_url(tmp_path):
    f = _write_json(tmp_path / "p.json", {"content": "<p>hi</p>", "encoding": "utf-8"})
    with pytest.raises(PageError, match="missing url"):
        load_page(f)


def test_load_page_missing_content(tmp_path):
    f = _write_json(tmp_path / "p.json", {"url": "https://a.com/", "encoding": "utf-8"})
    with pytest.raises(PageError, match="missing content"):
        load_page(f)


def test_load_page_empty_content_counts_as_missing(tmp_path):
    f = _write_json(tmp_path / "p.json",
                    {"url": "https://a.com/", "content": "", "encoding": "utf-8"})
    with pytest.raises(PageError, match="missing content"):
        load_page(f)


def test_load_page_uses_declared_encoding_for_non_utf8_bytes(tmp_path):
    # 0xE9 is "é" in ISO-8859-1 but is not valid UTF-8 on its own.
    f = tmp_path / "latin.json"
    f.write_bytes(b'{"url": "https://a.com/", "content": "caf\xe9 menu", '
                  b'"encoding": "ISO-8859-1"}')
    assert load_page(f) == ("https://a.com/", "café menu")


def test_load_page_replaces_undecodable_bytes_instead_of_failing(tmp_path):
    # Unknown declared encoding: falls back to UTF-8 with replacement characters.
    f = tmp_path / "weird.json"
    f.write_bytes(b'{"url": "https://a.com/", "content": "caf\xe9 menu", '
                  b'"encoding": "no-such-codec"}')
    url, content = load_page(f)
    assert url == "https://a.com/"
    assert content == "caf� menu"



def test_counts_plain_body_text():
    counts = extract_term_counts("<html><body><p>hello hello world</p></body></html>")
    assert counts == {"hello": (2, 0, 0, 0), "world": (1, 0, 0, 0)}


def test_script_style_noscript_and_comments_are_removed():
    html = ("<html><head><style>.cls { color: red }</style>"
            "<script>var secret = 1;</script></head>"
            "<body><noscript>enable javascript</noscript>"
            "<!-- hidden comment --><p>visible</p></body></html>")
    assert extract_term_counts(html) == {"visibl": (1, 0, 0, 0)}


def test_nav_sidebar_and_footer_text_is_kept():
    html = ("<body><nav>menu</nav><aside>sidebar</aside>"
            "<p>main</p><footer>contact</footer></body>")
    assert set(extract_term_counts(html)) == {"menu", "sidebar", "main", "contact"}


def test_neighboring_elements_do_not_merge():
    # Without a separator this would become the single token "ab".
    assert extract_term_counts("<p>a</p><p>b</p>") == {"a": (1, 0, 0, 0), "b": (1, 0, 0, 0)}


def test_broken_html_still_yields_tokens():
    counts = extract_term_counts("<html><body><p>Hello <b>world<p>again")
    assert {"hello", "world", "again"} <= set(counts)


def test_plain_text_without_tags_is_indexed():
    counts = extract_term_counts("BEGIN:VCALENDAR\r\nSUMMARY:Talk")
    assert {"begin", "vcalendar", "summari", "talk"} <= set(counts)


def test_nested_important_tags_count_once_as_the_strongest_kind():
    # Bold inside a heading counts as heading only, never as both.
    counts = extract_term_counts("<h1><strong>term</strong></h1>")
    assert counts["term"] == (1, 0, 1, 0)


def test_title_beats_heading_and_bold_when_nested():
    counts = extract_term_counts("<title><b>term</b></title>")
    assert counts["term"] == (1, 1, 0, 0)


def test_term_in_title_and_h1_counts_in_each():
    html = ("<html><head><title>term</title></head>"
            "<body><h1>term</h1></body></html>")
    assert extract_term_counts(html)["term"] == (2, 1, 1, 0)


def test_tf_includes_non_important_occurrences():
    html = ("<html><head><title>term</title></head>"
            "<body><h1>term</h1><p>term</p></body></html>")
    assert extract_term_counts(html)["term"] == (3, 1, 1, 0)


def test_each_important_tag_goes_to_its_kind_and_p_and_i_are_body():
    html = ("<b>bold</b><strong>strong</strong><h1>one</h1><h2>two</h2><h3>three</h3>"
            "<h4>four</h4><p>para</p><i>ital</i>")
    counts = extract_term_counts(html)
    assert counts["bold"] == (1, 0, 0, 1)
    assert counts["strong"] == (1, 0, 0, 1)
    assert counts["one"] == (1, 0, 1, 0)
    assert counts["two"] == (1, 0, 1, 0)
    assert counts["three"] == (1, 0, 1, 0)
    assert counts["four"] == (1, 0, 0, 0)
    assert counts["para"] == (1, 0, 0, 0)
    assert counts["ital"] == (1, 0, 0, 0)


def test_important_counts_never_exceed_tf():
    html = ("<html><head><title>Search Engine</title></head><body>"
            "<h1><b><strong>search</strong></b> engine</h1>"
            "<h2>engine <i>search</i></h2><p>search search engine</p>"
            "<strong><b>nested</b> nested</strong></body></html>")
    for term, (tf, title_tf, heading_tf, bold_tf) in extract_term_counts(html).items():
        assert min(title_tf, heading_tf, bold_tf) >= 0, term
        assert title_tf + heading_tf + bold_tf <= tf, term


def test_page_with_no_text_gives_empty_counts():
    assert extract_term_counts("<html><script>x = 1</script></html>") == {}


def test_script_nested_in_noscript_is_removed_without_error():
    html = "<noscript><script>hidden</script>also hidden</noscript><p>shown</p>"
    assert extract_term_counts(html) == {"shown": (1, 0, 0, 0)}


def test_load_page_declared_encoding_that_breaks_json_falls_back_to_utf8(tmp_path):
    # Not valid UTF-8, and re-decoding as UTF-16 would turn the JSON into garbage.
    f = tmp_path / "p.json"
    f.write_bytes(b'{"url": "https://a.com/", "content": "caf\xe9", "encoding": "utf-16"}')
    assert load_page(f) == ("https://a.com/", "caf�")


def test_load_page_read_error_becomes_page_error(tmp_path):
    # Reading a directory as a file raises an OSError; it must become a skip, not a crash.
    with pytest.raises(PageError, match="read error"):
        load_page(tmp_path)


def test_xml_looking_page_does_not_emit_warnings():
    xml = '<?xml version="1.0"?><rss><channel><title>feed</title></channel></rss>'
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert extract_term_counts(xml)["feed"] == (1, 1, 0, 0)



# ----- parse_page: counts plus word pairs, links, and a content fingerprint -----

def test_parse_page_counts_match_extract_term_counts():
    html = "<title>Search</title><h1>engine</h1><p>search <b>engine</b></p>"
    assert parse_page(html, "https://a.com/").counts == extract_term_counts(html)


def test_ngrams_are_neighboring_words_in_page_order():
    page = parse_page("<p>ICS 31 course</p><p>ICS 31</p>", "https://a.com/")
    assert page.ngram_counts == {"ic 31": 2, "31 cours": 1, "cours ic": 1,
                                 "ic 31 cours": 1, "31 cours ic": 1, "cours ic 31": 1}


def test_ngrams_join_words_split_by_inline_tags():
    # "ICS <b>31</b>" is two text nodes but one phrase.
    page = parse_page("<p>ICS <b>31</b></p>", "https://a.com/")
    assert page.ngram_counts == {"ic 31": 1}


def test_links_are_absolute_defragmented_with_anchor_words():
    html = '<a href="../grad/phd#top">PhD Programs</a><a href="https://b.com/">Apply Now</a>'
    page = parse_page(html, "https://a.com/dir/page")
    assert page.links == [("https://a.com/grad/phd", ["phd", "program"]),
                          ("https://b.com/", ["appli", "now"])]


def test_links_without_href_or_words_are_ignored():
    html = '<a name="top">Top</a><a href="/x"><img src="i.png"></a><a href="/y">  </a>'
    assert parse_page(html, "https://a.com/").links == []


def test_link_inside_removed_script_is_ignored():
    html = '<noscript><a href="/x">hidden</a></noscript><p>text</p>'
    assert parse_page(html, "https://a.com/").links == []


def test_same_words_give_same_fingerprint_even_with_different_tags():
    a = parse_page("<p>hello world</p>", "https://a.com/1")
    b = parse_page("<div>Hello</div><div>WORLD</div>", "https://a.com/2")
    assert a.fingerprint == b.fingerprint


def test_different_words_or_order_give_different_fingerprint():
    a = parse_page("<p>hello world</p>", "https://a.com/")
    assert a.fingerprint != parse_page("<p>hello there</p>", "https://a.com/").fingerprint
    assert a.fingerprint != parse_page("<p>world hello</p>", "https://a.com/").fingerprint


def test_malformed_link_is_skipped_instead_of_crashing():
    html = '<a href="http://[bad-ipv6/">broken</a><a href="/ok">fine</a>'
    assert parse_page(html, "https://a.com/").links == [("https://a.com/ok", ["fine"])]


def test_3gram_is_counted_for_a_three_word_phrase():
    page = parse_page("<p>How to apply</p>", "https://a.com/")
    assert page.ngram_counts == {"how to": 1, "to appli": 1, "how to appli": 1}

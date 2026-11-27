"""Tests for app.py: the search page, using Flask's test client (no server)."""
import pytest

from app import create_app


@pytest.fixture
def client(engine):
    return create_app(engine).test_client()


def test_home_shows_search_box_and_no_results(client):
    html = client.get("/").get_data(as_text=True)
    assert 'name="q"' in html
    assert "results for" not in html


def test_query_lists_urls_in_rank_order_with_timing(client):
    html = client.get("/?q=dog").get_data(as_text=True)
    assert 'href="u2"' in html and 'href="u1"' in html
    assert html.index('href="u2"') < html.index('href="u1"')
    assert "Top 2 results for" in html
    assert " ms" in html


def test_query_with_no_matches_says_so(client):
    html = client.get("/?q=zebra").get_data(as_text=True)
    assert "No results found." in html


def test_query_text_is_html_escaped(client):
    html = client.get("/?q=<script>alert(1)</script>").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html

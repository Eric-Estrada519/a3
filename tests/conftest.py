"""Shared tiny index for the search tests."""
import json

import pytest

from indexer import NGRAMS_FILE, DOC_IDS_FILE, INDEX_FILE


def _p(doc_id, tf, title_tf=0, heading_tf=0, bold_tf=0, anchor_tf=0):
    return {"doc_id": doc_id, "tf": tf, "title_tf": title_tf,
            "heading_tf": heading_tf, "bold_tf": bold_tf, "anchor_tf": anchor_tf}


TINY_INDEX = {
    "cat": [_p(0, 3), _p(1, 1)],
    "dog": [_p(1, 1), _p(2, 1, title_tf=1)],
    "the": [_p(0, 1), _p(1, 1), _p(2, 1), _p(3, 1)],
}
TINY_DOC_IDS = {"0": "u0", "1": "u1", "2": "u2", "3": "u3"}
TINY_NGRAMS = {"cat dog": [{"doc_id": 1, "tf": 1}]}


@pytest.fixture
def engine():
    from search import SearchEngine
    return SearchEngine(TINY_INDEX, TINY_DOC_IDS, TINY_NGRAMS)


@pytest.fixture
def index_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / INDEX_FILE).write_text(json.dumps(TINY_INDEX), encoding="utf-8")
    (out / DOC_IDS_FILE).write_text(json.dumps(TINY_DOC_IDS), encoding="utf-8")
    (out / NGRAMS_FILE).write_text(json.dumps(TINY_NGRAMS), encoding="utf-8")
    return out

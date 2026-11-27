"""Tests for evaluate.py: reading queries.txt, match counts, and the log file."""
from evaluate import count_matches, load_queries, main, run_evaluation


def test_load_queries_skips_comments_and_blank_lines(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text("# note\n\nGOOD | cat | expected | page\nBAD|dog|why\n", encoding="utf-8")
    assert load_queries(f) == [("GOOD", "cat", "expected | page"), ("BAD", "dog", "why")]


def test_count_matches_all_and_any(engine):
    assert count_matches(engine, "cat dog") == (1, 3)   # u1 has both; u0, u1, u2 have one
    assert count_matches(engine, "zebra") == (0, 0)
    assert count_matches(engine, "") == (0, 0)


def test_run_evaluation_lists_results_and_timing(engine):
    log = run_evaluation(engine, [("GOOD", "dog", "title match"), ("BAD", "zebra", "none")])
    assert "[GOOD] dog" in log and "all terms: 2, any term: 2" in log
    assert log.index("u2") < log.index("u1")
    assert "(no results)" in log
    assert "Slowest query:" in log


def test_main_writes_log(index_dir, tmp_path, capsys):
    queries = tmp_path / "q.txt"
    queries.write_text("GOOD | cat | test\n", encoding="utf-8")
    log = tmp_path / "log.txt"
    assert main([str(index_dir), str(log), "--queries", str(queries)]) == 0
    assert "[GOOD] cat" in log.read_text(encoding="utf-8")

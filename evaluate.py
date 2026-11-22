"""Run the test queries and log the results: python evaluate.py [output_dir] [log_file]"""
import argparse
import time
from pathlib import Path

from search import SearchEngine
from tokenizer import tokenize

# Number of results logged per query.
TOP_K = 5


# Read queries.txt lines of the form "TAG | query | reason". Lines starting with # are notes.
def load_queries(path) -> list[tuple[str, str, str]]:
    queries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tag, query, reason = (part.strip() for part in line.split("|", 2))
        queries.append((tag, query, reason))
    return queries


# Count pages that contain every query term, and pages that contain at least one.
def count_matches(engine: SearchEngine, query: str) -> tuple[int, int]:
    doc_sets = [{p["doc_id"] for p in engine.index.get(term, [])}
                for term in dict.fromkeys(tokenize(query))]
    if not doc_sets:
        return 0, 0
    return len(set.intersection(*doc_sets)), len(set.union(*doc_sets))


# Run every query, timing each search, and return the log text.
def run_evaluation(engine: SearchEngine, queries) -> str:
    lines = []
    slowest = 0.0
    for tag, query, reason in queries:
        start = time.perf_counter()
        results = engine.search(query, k=TOP_K)
        ms = (time.perf_counter() - start) * 1000
        slowest = max(slowest, ms)
        all_terms, any_term = count_matches(engine, query)
        lines.append(f"[{tag}] {query}  ({ms:.1f} ms, all terms: {all_terms}, any term: {any_term})")
        lines.append(f"    why: {reason}")
        for rank, r in enumerate(results, start=1):
            lines.append(f"    {rank}. {r.score:7.3f}  {r.url}")
        if not results:
            lines.append("    (no results)")
        lines.append("")
    lines.append(f"Slowest query: {slowest:.1f} ms")
    return "\n".join(lines) + "\n"


# Command line: load the index once, run queries.txt, write the log and print it.
def main(argv=None) -> int:
    arg_parser = argparse.ArgumentParser(description="Run the test queries.")
    arg_parser.add_argument("output_dir", nargs="?", default="output",
                            help="folder with the saved index (default: output)")
    arg_parser.add_argument("log_file", nargs="?", default="eval_log.txt",
                            help="where to write the results (default: eval_log.txt)")
    arg_parser.add_argument("--queries", default="queries.txt", help="query list file")
    args = arg_parser.parse_args(argv)

    engine = SearchEngine.from_dir(args.output_dir)
    log = run_evaluation(engine, load_queries(args.queries))
    Path(args.log_file).write_text(log, encoding="utf-8")
    print(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

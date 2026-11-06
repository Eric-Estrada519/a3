"""Write the Markdown report: python report.py <output_dir> <report.md>"""
import argparse
import json
from pathlib import Path

from indexer import DOC_IDS_FILE, INDEX_FILE

# Files counted in the "total index size on disk".
SIZE_FILES = (INDEX_FILE, DOC_IDS_FILE)


# Reload the saved index files and compute the report numbers.
def compute_stats(output_dir) -> dict:
    output_dir = Path(output_dir)
    index = json.loads((output_dir / INDEX_FILE).read_text(encoding="utf-8"))
    doc_ids = json.loads((output_dir / DOC_IDS_FILE).read_text(encoding="utf-8"))
    file_sizes = {name: (output_dir / name).stat().st_size for name in SIZE_FILES}
    return {
        "num_documents": len(doc_ids),
        "num_unique_tokens": len(index),
        "file_sizes": file_sizes,
        "total_size_kb": round(sum(file_sizes.values()) / 1024, 2),
    }


# Turn the stats into the report's Markdown text.
def render_markdown(stats: dict) -> str:
    total_bytes = sum(stats["file_sizes"].values())
    file_rows = "\n".join(
        f"| `{name}` | {size} |" for name, size in stats["file_sizes"].items()
    )
    return f"""# Assignment 3, Milestone 1: Index Report

**Group members:** Eric Estrada 87128480

**Course / quarter:** Fall 2026

**Track:** Information Analyst

## Index analytics

| Metric | Value |
|---|---|
| Number of indexed documents | {stats["num_documents"]} |
| Number of unique tokens | {stats["num_unique_tokens"]} |
| Total index size on disk (KB) | {stats["total_size_kb"]:.2f} |

Unique tokens are the distinct Porter-stemmed terms in the inverted index.

## Files counted in the index size

| File | Size (bytes) |
|---|---|
{file_rows}

Total: {total_bytes} bytes / 1024 = {stats["total_size_kb"]:.2f} KB
"""


# compute the stats and write the report file all in the command line.
def main(argv=None) -> int:
    arg_parser = argparse.ArgumentParser(description="Write the index report.")
    arg_parser.add_argument("output_dir", help="folder with the saved index")
    arg_parser.add_argument("report_path", help="report .md file to write")
    args = arg_parser.parse_args(argv)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(compute_stats(args.output_dir)), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

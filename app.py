"""Search web page: python app.py <output_dir>, then open http://127.0.0.1:5000"""
import argparse
import time

from flask import Flask, render_template_string, request

from search import SearchEngine

# Number of results shown per query.
TOP_K = 5

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ICS Search</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
    form { display: flex; gap: 8px; }
    input[type=text] { flex: 1; padding: 8px; font-size: 16px; }
    button { padding: 8px 16px; font-size: 16px; }
    .meta { color: #555; margin: 16px 0 8px; }
    li { margin-bottom: 10px; }
    .score { color: #777; font-size: 13px; }
  </style>
</head>
<body>
  <h1>ICS Search</h1>
  <form action="/" method="get">
    <input type="text" name="q" value="{{ query }}" placeholder="Search UCI ICS pages" autofocus>
    <button type="submit">Search</button>
  </form>
  {% if query %}
    {% if results %}
      <p class="meta">Top {{ results|length }} results for "{{ query }}" ({{ "%.1f"|format(ms) }} ms)</p>
      <ol>
      {% for r in results %}
        <li><a href="{{ r.url }}">{{ r.url }}</a> <span class="score">score {{ "%.3f"|format(r.score) }}</span></li>
      {% endfor %}
      </ol>
    {% else %}
      <p class="meta">No results found. ({{ "%.1f"|format(ms) }} ms)</p>
    {% endif %}
  {% endif %}
</body>
</html>
"""


# Build the Flask app around an already-loaded search engine.
def create_app(engine: SearchEngine) -> Flask:
    app = Flask(__name__)

    # Show the search box and, if a query was given, the top results and timing.
    @app.route("/")
    def home():
        query = request.args.get("q", "").strip()
        results, ms = [], 0.0
        if query:
            start = time.perf_counter()
            results = engine.search(query, k=TOP_K)
            ms = (time.perf_counter() - start) * 1000
        return render_template_string(PAGE, query=query, results=results, ms=ms)

    return app


# Command line: load the index once, then serve the page on this computer only.
def main(argv=None) -> None:
    arg_parser = argparse.ArgumentParser(description="Run the search web page.")
    arg_parser.add_argument("output_dir", help="folder with the saved index")
    args = arg_parser.parse_args(argv)
    app = create_app(SearchEngine.from_dir(args.output_dir))
    app.run(host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()

ICS Search Engine - Assignment 3
================================

Track:          Information Analyst
Group members:  Eric Estrada 87128480
Course:         Fall 2026


WHAT IS IN THIS ZIP
-------------------

  tokenizer.py      splits text into lowercase alphanumeric words and Porter-stems them
                    (used by both the indexer and the search, so they always match)
  parser.py         reads one JSON page, removes the URL #fragment, and counts each
                    word in the body, title, headings (h1-h3) and bold text
  indexer.py        builds the inverted index from the dataset folder and saves it
  search.py         ranked search over the saved index, with a console interface
  app.py            the same search as a local web page (Flask)
  evaluate.py       runs every test query in queries.txt and logs the top 5 results
  queries.txt       the test queries (see TEST.txt for the explanation)
  report.py         writes the Milestone 1 analytics report (optional)
  report_m2.py      writes the Milestone 2 query report (optional)
  tests/            pytest unit tests (optional)
  requirements.txt  Python packages needed


1. SETUP
--------

Needs Python 3.9 or newer. From the project folder, install the packages:

    pip install -r requirements.txt

Put the dataset in the project folder so that the page folders are inside
analyst/ANALYST/ (unzip analyst.zip here). The layout should look like:

    analyst/ANALYST/<domain folder>/<page>.json

If the dataset is somewhere else, pass its path to indexer.py (see step 2).


2. BUILD THE INDEX
------------------

    python indexer.py

This reads every JSON page under analyst/ANALYST and writes the index to the
folder "output". It takes about a minute. To use other folders:

    python indexer.py <dataset_folder> <output_folder>

It prints "Indexed N of M files" when done. Files written to the output folder:

    index.json     the inverted index: term -> list of postings
                   {doc_id, tf, title_tf, heading_tf, bold_tf}
    doc_ids.json   doc ID -> URL
    skipped.json   files that were not indexed and why (duplicate URL, no text)

The index is not included in this zip. Running this step recreates it.


3. START THE SEARCH INTERFACE
-----------------------------

Option A - console (text) interface:

    python search.py

(or "python search.py <output_folder>" if the index is not in "output")
It loads the index once (a few seconds), then shows a prompt.

Option B - web interface:

    python app.py output

then open http://127.0.0.1:5000 in a browser. Here the output folder argument
is required. Press Ctrl+C in the terminal to stop the server.


4. PERFORM A SIMPLE QUERY
-------------------------

Console: type a query at the prompt and press Enter, for example

    Search (or 'quit'): machine learning

It prints the top 10 URLs with their scores and the time the search took.
Type another query, or "quit" (or press Ctrl+C) to exit.

Web: type a query in the search box and press Search. The top 5 URLs are
shown with their scores and the time taken.

Queries are case-insensitive and stemmed the same way as the index, so
"running" also matches "run" and "runs". Pages that contain more of the
query words are ranked first, then by score.


5. RUN THE TEST QUERIES (OPTIONAL)
----------------------------------

    python evaluate.py

Runs all queries in queries.txt and writes the top 5 results and the time of
each query to eval_log.txt.

Unit tests:

    python -m pytest

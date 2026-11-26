ICS Search Engine - Assignment 3
================================

Track:          Information Analyst
Group members:  Eric Estrada 87128480



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
  requirements.txt  Python packages needed
  tests/            automated tests for every part (see step 6)
  pytest.ini        settings for the tests


1. SETUP
--------

Needs Python 3.10 or newer. From the project folder, install the packages:

    pip install -r requirements.txt

Unzip analyst.zip into the project folder. It contains one folder, ANALYST,
with one folder per domain inside it:

    ANALYST/<domain folder>/<page>.json


2. BUILD THE INDEX
------------------

    python indexer.py ANALYST

This reads every JSON page under the ANALYST folder and writes the index to
the folder "output". It takes about a minute. If the dataset is somewhere
else, give its path instead of ANALYST. To also choose the output folder:

    python indexer.py <dataset_folder> <output_folder>

(With no arguments, the dataset folder defaults to analyst/ANALYST.)

It prints "Indexed N of M files" when done. Files written to the output folder:

    index.json     the inverted index: term -> list of postings
                   {doc_id, tf, title_tf, heading_tf, bold_tf, anchor_tf}
    doc_ids.json   doc ID -> URL
    ngrams.json    2- and 3-word phrases: "word1 word2" or "word1 word2 word3"
                   -> list of {doc_id, tf}
    pagerank.json  doc ID -> PageRank (how well linked the page is; sums to 1)
    skipped.json   files that were not indexed and why (duplicate URL,
                   duplicate content, no text)

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


6. RUN THE AUTOMATED TESTS (OPTIONAL)
-------------------------------------

    python -m pytest

Runs the tests in tests/. They build small indexes from made-up pages, so
most of them do not need the dataset. The tests in tests/test_real_index.py
check the real index in "output" against the dataset in analyst/ANALYST, and
are skipped if either is missing. To point them at another dataset folder,
set the A3_DATASET_DIR environment variable to its path first.

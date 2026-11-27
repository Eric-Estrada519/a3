"""Tests for tokenizer.tokenize: regex split, lowercasing, Porter stemming."""
from tokenizer import ngrams, tokenize


def test_hyphenated_word_splits_and_stems():
    # "Self-Driving" -> "self", "driving" -> stemmed "drive"
    assert tokenize("Self-Driving") == ["self", "drive"]


def test_lowercases():
    assert tokenize("HELLO World") == ["hello", "world"]


def test_digits_are_kept_inside_tokens():
    assert tokenize("CS121 and 2024") == ["cs121", "and", "2024"]


def test_non_ascii_characters_split_tokens():
    # Only [a-zA-Z0-9] count, so the "ï" acts as a separator.
    assert tokenize("naïve") == ["na", "ve"]


def test_stop_words_are_kept():
    assert tokenize("the of a") == ["the", "of", "a"]


def test_porter_stemming_is_applied():
    assert tokenize("running computers computing") == ["run", "comput", "comput"]


def test_empty_and_punctuation_only_give_no_tokens():
    assert tokenize("") == []
    assert tokenize("!!! --- ...") == []


def test_dotted_abbreviation_also_gives_joined_word():
    # "Ph.D." keeps "ph" and "d" (every alphanumeric run is a token) and adds "phd".
    assert tokenize("Ph.D. program") == ["ph", "d", "phd", "program"]


def test_single_letter_abbreviations_are_joined():
    assert tokenize("the U.S. and U.S.A.") == ["the", "u", "s", "us", "and", "u", "s", "a", "usa"]


def test_query_phd_matches_joined_word_from_page():
    assert "phd" in tokenize("PhD") and "phd" in tokenize("Ph.D.")


def test_web_addresses_and_numbers_are_not_joined():
    # Pieces longer than two letters, or digits, are not abbreviations.
    assert tokenize("www.ics.uci.edu") == ["www", "ic", "uci", "edu"]
    assert tokenize("version 3.14") == ["version", "3", "14"]
    assert tokenize("St.Louis") == ["st", "loui"]


def test_initial_before_a_name_is_not_joined():
    # "V. Lopes" is an initial followed by a space, not an abbreviation.
    assert tokenize("Cristina V. Lopes") == ["cristina", "v", "lope"]


def test_2grams_pair_each_token_with_the_next():
    assert ngrams(["ic", "31", "cours"], 2) == ["ic 31", "31 cours"]


def test_3grams_join_each_run_of_three_tokens():
    assert ngrams(["how", "to", "appli", "now"], 3) == ["how to appli", "to appli now"]


def test_ngrams_need_at_least_n_tokens():
    assert ngrams(["ic"], 2) == []
    assert ngrams(["how", "to"], 3) == []
    assert ngrams([], 2) == []

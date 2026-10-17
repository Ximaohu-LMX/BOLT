import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from src.LPLM.extension import EscChar, esc_list
from src.LPLM.misc_utils import get_vocab_from_list, loadtestQueries


class LPLMEncodingTest(unittest.TestCase):
    def test_wildcard_escape_tokens_are_unique_and_round_trip(self):
        self.assertEqual(len(esc_list), len(set(esc_list)))
        for n_under in range(1, 11):
            for has_percent in (False, True):
                wild = ("%" if has_percent else "") + "_" * n_under
                token = EscChar.getEscChar(wild)
                decoded = EscChar.getWildSequence(token)
                self.assertEqual(decoded.count("_"), n_under)
                self.assertEqual("%" in decoded, has_percent)

    def test_vocab_reserves_every_structural_token(self):
        with contextlib.redirect_stdout(io.StringIO()):
            vocab = get_vocab_from_list(["abc"], max_n_char=200)
        self.assertTrue(set(esc_list) <= set(vocab))

    def test_long_wildcard_queries_are_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "test.txt"
            queries = ["%Nat_____%t%", "%hris_oph%r ______in_%"]
            path.write_text("\n".join(f"{q},1" for q in queries) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                vocab = get_vocab_from_list(["Natthrisopn "], max_n_char=200)
            char2idx = {char: i for i, char in enumerate(vocab)}
            loaded, cards = loadtestQueries(path, char2idx)
            self.assertEqual(loaded, queries)
            self.assertEqual(cards, [1, 1])

    def test_unsupported_literal_fails_instead_of_dropping_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text("abcΩ,1\n")
            char2idx = {char: i for i, char in enumerate("abc" + "".join(esc_list))}
            with self.assertRaisesRegex(ValueError, "U\\+03A9"):
                loadtestQueries(path, char2idx)


if __name__ == "__main__":
    unittest.main()

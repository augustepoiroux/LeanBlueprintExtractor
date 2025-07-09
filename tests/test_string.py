import unittest

from lean_interact.utils import compress_newlines


class TestCompressNewlines(unittest.TestCase):
    def test_compress_newlines_multiple_newlines(self):
        input_code = "line1\n\n\n\nline2"
        expected_output = "line1\n\nline2"
        self.assertEqual(compress_newlines(input_code), expected_output)

    def test_compress_newlines_whitespace_lines(self):
        input_code = "line1\n  \n \nline2"
        expected_output = "line1\n\nline2"
        self.assertEqual(compress_newlines(input_code), expected_output)

    def test_compress_newlines_no_compression_needed(self):
        input_code = "line1\n\nline2"
        expected_output = "line1\n\nline2"
        self.assertEqual(compress_newlines(input_code), expected_output)

    def test_compress_newlines_empty_string(self):
        input_code = ""
        expected_output = ""
        self.assertEqual(compress_newlines(input_code), expected_output)

    def test_compress_newlines_basic(self):
        input_code = (
            "/- This is a comment -/\n  \n \ntheorem my_theorem : True := by\n-- single line comment\n  sorry\n"
        )
        expected_output = (
            "/- This is a comment -/\n\ntheorem my_theorem : True := by\n-- single line comment\n  sorry\n"
        )
        self.assertEqual(compress_newlines(input_code), expected_output)


if __name__ == "__main__":
    unittest.main()

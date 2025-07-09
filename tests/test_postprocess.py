import unittest

from lean_blueprint_extractor.utils import extract_lean_codes


class TestExtractLeanCodes(unittest.TestCase):
    def test_no_lean_code(self):
        content = "Some regular text without code."
        expected = ["Some regular text without code."]
        self.assertEqual(extract_lean_codes(content), expected)

    def test_single_lean_code_block(self):
        content = "Here is some text.\n```lean4\ndef foo : Nat := 42\n```\nMore text."
        expected = ["def foo : Nat := 42"]
        self.assertEqual(extract_lean_codes(content), expected)

    def test_multiple_lean_code_blocks(self):
        content = """Text before.

```lean4
def foo : Nat := 42
```

Some other text.

```lean4
def bar : Nat := foo + 1
```
"""
        expected = ["def foo : Nat := 42", "def bar : Nat := foo + 1"]
        self.assertEqual(extract_lean_codes(content), expected)

    def test_incomplete_code_block(self):
        content = "Text before.\n```lean4\ndef foo : Nat := 42\nMore text without closing code block."
        expected = ["def foo : Nat := 42\nMore text without closing code block."]
        self.assertEqual(extract_lean_codes(content), expected)

    def test_empty_code_block(self):
        content = "```lean4\n```\nSome text."
        expected = [""]
        self.assertEqual(extract_lean_codes(content), expected)

    def test_code_block_with_lean_marker_no_4(self):
        content = """Here is a test:
```lean
def basic : Nat := 0
```
End."""
        expected = ["def basic : Nat := 0"]
        self.assertEqual(extract_lean_codes(content), expected)

    def test_code_block_with_lean_marker_and_spaces(self):
        content = """```
 Some text that is not Lean
```
```lean
theorem test : Bool := true
```
"""
        expected = ["theorem test : Bool := true"]
        self.assertEqual(extract_lean_codes(content), expected)


if __name__ == "__main__":
    unittest.main()

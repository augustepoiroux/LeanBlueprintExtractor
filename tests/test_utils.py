import unittest

from lean_interact.utils import (
    clean_last_theorem_string,
    clean_theorem_string,
    extract_last_theorem,
    indent_code,
    remove_lean_comments,
    split_conclusion,
    split_implementation,
)

from lean_blueprint_extractor.utils import (
    LeanFilesProcessor,
    lean_comments_ranges,
    replace_ranges,
    trim_comments_end,
)


class TestLeanUtils(unittest.TestCase):
    def test_indent_code(self):
        code = "theorem my_theorem : True := by\n  sorry"
        expected = "  theorem my_theorem : True := by\n    sorry"
        self.assertEqual(indent_code(code, 2), expected)

    def test_lean_comments_ranges(self):
        code = "/- This is a comment -/\ntheorem my_theorem : True := by\n-- single line comment\n  sorry"
        comment_ranges = lean_comments_ranges(code)
        expected_first_comment = "/- This is a comment -/"
        expected_second_comment = "-- single line comment"
        self.assertEqual(code[comment_ranges[0][0] : comment_ranges[0][1]], expected_first_comment)
        self.assertEqual(code[comment_ranges[1][0] : comment_ranges[1][1]], expected_second_comment)

    def test_remove_lean_comments(self):
        code = "/- This is a comment -/\ntheorem my_theorem : True := by\n-- single line comment\n  sorry"
        expected = "\ntheorem my_theorem : True := by\n\n  sorry"
        self.assertEqual(remove_lean_comments(code), expected)

    def test_remove_lean_comments_non_opened(self):
        code = " This is a comment -/\ntheorem my_theorem : True := by\n-- single line comment\n  sorry"
        expected = "\ntheorem my_theorem : True := by\n\n  sorry"
        self.assertEqual(remove_lean_comments(code), expected)

    def test_split_implementation(self):
        declaration = "theorem my_theorem : True := by\n  sorry"
        split_idx = split_implementation(declaration)
        expected_first = "theorem my_theorem : True "
        expected_second = ":= by\n  sorry"
        self.assertEqual(declaration[:split_idx], expected_first)
        self.assertEqual(declaration[split_idx:], expected_second)

    def test_split_conclusion(self):
        declaration = "theorem my_theorem : True := by\n  sorry"
        split_idx = split_conclusion(declaration)
        expected_first = "theorem my_theorem "
        expected_second = ": True := by\n  sorry"
        self.assertEqual(declaration[:split_idx], expected_first)
        self.assertEqual(declaration[split_idx:], expected_second)

    def test_clean_theorem_string(self):
        theorem_string = "/- comment -/ theorem my_theorem  : \nTrue := by\n  sorry"
        expected = "theorem dummy : \nTrue"
        self.assertEqual(clean_theorem_string(theorem_string, "dummy", add_sorry=False), expected)

    def test_clean_theorem_string_partial_comment_1(self):
        theorem_string = "comment -/ theorem my_theorem  : \nTrue := by\n  sorry"
        expected = "theorem dummy : \nTrue"
        self.assertEqual(clean_theorem_string(theorem_string, "dummy", add_sorry=False), expected)

    def test_clean_theorem_string_partial_comment_2(self):
        theorem_string = """
lemma integrable_x_mul_Smooth1 (SmoothingF : ℝ → ℝ) (ε : ℝ) (hF : ∀ x, 0 ≤ SmoothingF x) (hF' : ContinuousWithDeriv (fun x ↦ SmoothingF x / x) ℝ) (hsupp :
support (fun x ↦ SmoothingF x) = Icc (1/2) 2) (htotal : ∫ x : ℝ in Ioi 0, SmoothingF x / x) = 1) :
  IntegrableOn (fun x : ℝ ↦ x * SmoothingF x) (Ioi 0) :=
  sorry -/"""
        self.assertIsNone(clean_theorem_string(theorem_string, "integrable_x_mul_Smooth1", add_sorry=False))

    def test_clean_theorem_string_hard(self):
        theorem_string = "theorem test (h := 1): let x := 1; x = 1 := by\n  sorry"
        expected = "theorem dummy (h := 1): let x := 1; x = 1"
        self.assertEqual(clean_theorem_string(theorem_string, "dummy", add_sorry=False), expected)

    def test_clean_theorem_string_hard_real(self):
        theorem_string = """theorem nrRealPlaces_eq_zero [IsCyclotomicExtension {n} ℚ K]
  (hn : 2 < n) :
  haveI := IsCyclotomicExtension.numberField {n} ℚ K
  NrRealPlaces K = 0 := by
have := IsCyclotomicExtension.numberField {n} ℚ K
apply (IsCyclotomicExtension.zeta_spec n ℚ K).nrRealPlaces_eq_zero_of_two_lt h"""
        expected = """theorem dummy [IsCyclotomicExtension {n} ℚ K]
  (hn : 2 < n) :
  haveI := IsCyclotomicExtension.numberField {n} ℚ K
  NrRealPlaces K = 0"""
        self.assertEqual(clean_theorem_string(theorem_string, "dummy", add_sorry=False), expected)

    def test_clean_theorem_string_real_edge_case(self):
        theorem_string = """lemma ZetaNear1BndExact:
    ∃ (c : ℝ) (_ : 0 < c), ∀ (σ : ℝ) (_ : σ ∈ Ioc 1 2), ‖ζ σ‖ ≤ c / (σ - 1) := by sorry"""
        expected = """theorem dummy :
    ∃ (c : ℝ) (_ : 0 < c), ∀ (σ : ℝ) (_ : σ ∈ Ioc 1 2), ‖ζ σ‖ ≤ c / (σ - 1)"""
        self.assertEqual(clean_theorem_string(theorem_string, "dummy", add_sorry=False), expected)

    def test_extract_last_theorem(self):
        code = "/- comment -/ theorem my_theorem : True := by\n  sorry"
        start_idx = extract_last_theorem(code)
        expected_output = "theorem my_theorem : True := by\n  sorry"
        self.assertEqual(code[start_idx:], expected_output)

    def test_clean_last_theorem_string(self):
        code = "/- comment -/\ntheorem my_theorem : True := by\n  sorry"
        expected = "/- comment -/\ntheorem dummy : True"
        self.assertEqual(clean_last_theorem_string(code, "dummy"), expected)

    def test_lean_file_processor_replace_ranges(self):
        content = "theorem my_theorem : True := by\n  sorry"
        ranges = [(26, 40, ":= sorry")]
        expected = "theorem my_theorem : True := sorry"
        self.assertEqual(replace_ranges(content, ranges), expected)

    def test_lean_file_processor_remove_theorems(self):
        declarations = [
            {"file": "dummy.lean", "start_idx": 14, "end_idx": 53, "theorem_info": {}, "full_name": "my_theorem"}
        ]
        processor = LeanFilesProcessor(declarations)
        content = "/- comment -/\ntheorem my_theorem : True := by\n  sorry\n\n/- second comment -/\n"
        expected = "/- comment -/\n\n\n/- second comment -/\n"
        self.assertEqual(processor.remove_theorems("dummy.lean", content, compress_newlines_comments=False), expected)

    def test_lean_file_processor_remove_theorems_special_comments(self):
        declarations = [
            {"file": "dummy.lean", "start_idx": 14, "end_idx": 53, "theorem_info": {}, "full_name": "my_theorem"}
        ]
        processor = LeanFilesProcessor(declarations)
        content = (
            "/- comment -/\ntheorem my_theorem : True := by\n  sorry\n\n/-%% LaTeX comment %%-//- second comment -/\n"
        )
        expected = "/- comment -/\n\n\n/-%% LaTeX comment %%-//- second comment -/\n"
        self.assertEqual(processor.remove_theorems("dummy.lean", content, compress_newlines_comments=False), expected)

    def test_lean_file_processor_remove_theorems_compress(self):
        declarations = [
            {"file": "dummy.lean", "start_idx": 18, "end_idx": 57, "theorem_info": {}, "full_name": "my_theorem"}
        ]
        processor = LeanFilesProcessor(declarations)
        content = "/- top comment -/\ntheorem my_theorem : True := by\n  sorry\n\n/- second comment -/\n"
        expected = "/- second comment -/\n"
        self.assertEqual(processor.remove_theorems("dummy.lean", content), expected)

    def test_lean_file_processor_remove_proofs(self):
        declarations = [
            {
                "file": "test.lean",
                "start_idx": 14,
                "end_idx": 53,
                "theorem_info": {"proof_start_idx": 39},
                "full_name": "my_theorem",
            }
        ]
        processor = LeanFilesProcessor(declarations)
        content = "/- comment -/\ntheorem my_theorem : True := by\n  sorry\n\n/- second comment -/\n"
        expected = "/- comment -/\ntheorem my_theorem : True := sorry\n\n/- second comment -/\n"
        self.assertEqual(processor.remove_proofs("test.lean", content, compress_newlines=False), expected)

    def test_lean_file_processor_remove_proofs_compress(self):
        declarations = [
            {
                "file": "test.lean",
                "start_idx": 21,
                "end_idx": 60,
                "theorem_info": {"proof_start_idx": 46},
                "full_name": "my_theorem",
            }
        ]
        processor = LeanFilesProcessor(declarations)
        content = "/- top \ncomment\n\n -/\ntheorem my_theorem : True := by\n  sorry\n\n/- second comment -/\n"
        expected = "theorem my_theorem : True := sorry\n\n/- second comment -/\n"
        self.assertEqual(processor.remove_proofs("test.lean", content), expected)

    def test_trim_comments_end(self):
        code = "theorem my_theorem : True := by\n  sorry\n\n/- This is a comment -/\n\n"
        expected = "theorem my_theorem : True := by\n  sorry\n\n"
        self.assertEqual(trim_comments_end(code), expected)

    def test_trim_comments_end_with_whitespace(self):
        code = "theorem my_theorem : True := by\n  sorry\n    \n"
        expected = "theorem my_theorem : True := by\n  sorry\n"
        self.assertEqual(trim_comments_end(code), expected)

    def test_trim_comments_end_with_multiple_comments(self):
        code = "theorem my_theorem : True := by\n  sorry\n\n/- First comment -/\n\n/- Second comment -/\n"
        expected = "theorem my_theorem : True := by\n  sorry\n\n"
        self.assertEqual(trim_comments_end(code), expected)

    def test_trim_comments_end_no_comments(self):
        code = "theorem my_theorem : True := by\n  sorry"
        expected = code
        self.assertEqual(trim_comments_end(code), expected)

    def test_trim_comments_end_only_comments(self):
        code = "/- This is a comment -/\n\n/- Another comment -/\n   "
        expected = ""
        self.assertEqual(trim_comments_end(code), expected)

    def test_trim_comments_end_comments_at_beginning(self):
        code = "/- Comment at the beginning -/\ntheorem my_theorem : True := by\n  sorry\n\n/- End comment -/\n"
        expected = "/- Comment at the beginning -/\ntheorem my_theorem : True := by\n  sorry\n\n"
        self.assertEqual(trim_comments_end(code), expected)

    def test_trim_comments_end_empty_string(self):
        code = ""
        expected = ""
        self.assertEqual(trim_comments_end(code), expected)


if __name__ == "__main__":
    unittest.main()

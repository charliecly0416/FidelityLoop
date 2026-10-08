import unittest
from count_footprint import calculate, effective_lines


class CounterTests(unittest.TestCase):
    def test_docstrings_comments_and_multiline(self):
        text = '''"""module docs
continued"""
# comment
@decorate
def f():
    """function docs"""
    value = (
        1  # inline comment
        + 2
    )
    return value
'''
        self.assertEqual(effective_lines(text), {4, 5, 7, 8, 9, 10, 11})

    def test_string_data_is_code(self):
        text = 'value = """data\nmore data\n"""\n'
        self.assertEqual(effective_lines(text), {1, 2, 3})

    def test_semicolons_not_double_counted(self):
        self.assertEqual(effective_lines('a = 1; b = 2 # c\n'), {1})

    def test_complete_snapshot(self):
        result = calculate()
        self.assertGreater(result['partitioned_effective_physical_lines'], 500)
        self.assertIsNone(result['historical_change_evidence']['backend_delta'])


if __name__ == '__main__':
    unittest.main()

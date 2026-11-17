import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import inventory  # noqa: E402


class Inventory(unittest.TestCase):
    def test_parse_line(self):
        self.assertEqual(inventory.parse_line({'name': 'bolt', 'qty': '3', 'price': '0.25'}), ('bolt', 3, 0.25))

    def test_total_value(self):
        items = [('a', 2, 1.50), ('b', 1, 4.00), ('c', 10, 0.10)]
        self.assertEqual(inventory.total_value(items), 8.00)


if __name__ == '__main__':
    unittest.main()

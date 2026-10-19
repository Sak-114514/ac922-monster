"""The acceptance socket adds no implicit refill or cross-wave close command."""
import unittest

from candidate_server import close_ack, requested_refill


class PrivateRefillTests(unittest.TestCase):
    def test_default_and_explicit_boolean(self):
        self.assertIs(requested_refill({}), False)
        self.assertIs(requested_refill({"refill": False}), False)
        self.assertIs(requested_refill({"refill": True}), True)

    def test_other_types_rejected(self):
        for value in [None, 0, 1, "true", [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                requested_refill({"refill": value})

    def test_matching_close_only(self):
        self.assertEqual(close_ack({"id": "wave-1"}, "wave-1", True),
                         {"type": "close_wave", "id": "wave-1"})
        for event, refill in [({"id": "other"}, True), ({"id": "wave-1"}, False)]:
            with self.assertRaises(RuntimeError):
                close_ack(event, "wave-1", refill)


if __name__ == "__main__":
    unittest.main()

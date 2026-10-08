import unittest
from unittest.mock import patch

import numpy as np

from config import Config
from rotation import evaluate


class SingleIocTests(unittest.TestCase):
    def test_default_uses_only_trailing_nine_ms(self):
        events = np.array([[1, 1, t, 1] for t in (.020, .0243329, .024333, .030, .0333329, .033333)])
        received = []

        def capture(local, box, end, cfg):
            received.extend(local[:, 2].tolist())
            return {'rotation_q': 0.0, 'valid_pixels': 0, 'observable': False}

        with patch('rotation.evaluate_slice', side_effect=capture):
            evaluate(events, (0, 0, 4, 4), .033333, Config(), start=0)
        self.assertEqual(received, [.024333, .030, .0333329])

    def test_duration_can_cover_complete_detection_window(self):
        events = np.array([[1, 1, t, 1] for t in (0.0, .01, .0333329)])
        received = []

        def capture(local, box, end, cfg):
            received.extend(local[:, 2].tolist())
            return {'rotation_q': 0.0, 'valid_pixels': 0, 'observable': False}

        cfg = Config(rotation_window_ms=33.333)
        with patch('rotation.evaluate_slice', side_effect=capture):
            evaluate(events, (0, 0, 4, 4), .033333, cfg, start=0)
        self.assertEqual(received, [0.0, .01, .0333329])

    def test_rotation_interval_cannot_exceed_detection_window(self):
        with self.assertRaises(ValueError):
            Config(rotation_window_ms=34).validate()


if __name__ == '__main__':
    unittest.main()

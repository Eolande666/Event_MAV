import math
import unittest
from persistence.types import Detection
from persistence.neighborhood import neighborhood_score
from persistence.trajectory import motion_features
from persistence.scoring import trajectory_score, fuse_scores


def obs(t, x=0, y=0, size=2):
    return Detection(t, (x-size/2, y-size/2, x+size/2, y+size/2), 500, 6, 10)


def motion(history, current):
    return motion_features(history, current, min_speed=.5, epsilon=1e-9)


def score(m):
    return trajectory_score(m, weights=dict(position=.5, direction=.25, acceleration=.25),
                            sigmas=dict(position=1, direction=1, acceleration=1), normalize_position=True)


class PersistenceTests(unittest.TestCase):
    def test_disabled_config_needs_no_scales(self):
        from persistence.config import Config
        Config().validate()
        with self.assertRaises(ValueError):
            Config(enabled=True).validate()

    def test_warmup(self):
        self.assertIsNone(motion([], obs(0)).speed)
        m = motion([obs(0)], obs(1, 2))
        self.assertEqual(m.velocity, (2, 0))
        self.assertIsNone(m.prediction_error)
        self.assertIsNone(score(m)[0])

    def test_stationary_disables_direction(self):
        m = motion([obs(0), obs(1)], obs(2))
        self.assertIsNone(m.direction_change)
        value, weights = score(m)
        self.assertEqual(value, 1)
        self.assertNotIn('direction', weights)
        self.assertAlmostEqual(sum(weights.values()), 1)

    def test_irregular_constant_velocity(self):
        m = motion([obs(0), obs(.3, .6)], obs(1.1, 2.2))
        self.assertAlmostEqual(m.prediction_error, 0)
        self.assertAlmostEqual(m.acceleration_norm, 0)
        self.assertAlmostEqual(score(m)[0], 1, places=8)

    def test_current_does_not_leak_into_prediction(self):
        m = motion([obs(0), obs(1, 1)], obs(2, 100))
        self.assertEqual(m.predicted_center, (2, 0))
        self.assertEqual(m.prediction_error, 98)
        self.assertLess(score(m)[0], .01)

    def test_turn_soft_penalty(self):
        history = [obs(0), obs(1, 1)]
        mild = score(motion(history, obs(2, 2, .1)))[0]
        sharp = score(motion(history, obs(2, 1, 1)))[0]
        self.assertGreater(mild, sharp)
        self.assertGreater(sharp, 0)

    def test_invalid_times(self):
        for t in (0, -1):
            with self.assertRaises(ValueError):
                motion([obs(0)], obs(t))

    def test_invalid_observation(self):
        for x in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                obs(0, x)
        with self.assertRaises(ValueError):
            obs(0, size=0)
        self.assertTrue(math.isfinite(motion([obs(0, size=1e-6), obs(1, size=1e-6)], obs(2, 1, size=1e-6)).normalized_error))

    def test_neighborhood(self):
        def run(hits, **kw):
            return neighborhood_score(hits, length=4, mode=kw.get('mode','weighted'), decay=kw.get('decay',1), cold_start=kw.get('cold_start','fixed'))
        self.assertEqual(run([1]), .25)
        self.assertEqual(run([1], cold_start='observed'), 1)
        self.assertIsNone(run([]))
        self.assertEqual(run([1,1,0,1]), .75)
        self.assertEqual(run([0,1,1,0,1]), .75)
        self.assertAlmostEqual(run([1,0,1,1], decay=.5), (1+.5+.125)/1.875)
        self.assertEqual(run([1,0,1], mode='binary', decay=.5), run([1,0,1]))

    def test_missing_scale_is_error(self):
        with self.assertRaises(ValueError):
            trajectory_score(motion([], obs(0)), weights=dict(position=1,direction=0,acceleration=0),
                             sigmas=dict(position=None,direction=None,acceleration=None), normalize_position=True)

    def test_fusion_missing(self):
        self.assertEqual(fuse_scores(1, None, alpha=.5), 1)
        self.assertEqual(fuse_scores(1, None, alpha=1), 1)
        self.assertEqual(fuse_scores(1, None, alpha=0), 1)
        self.assertAlmostEqual(fuse_scores(.4,.8,alpha=.5), .6)
        with self.assertRaises(ValueError):
            fuse_scores(math.nan, 1, alpha=.5)


if __name__ == '__main__':
    unittest.main()

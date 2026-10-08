"""Behavior tests for scoring, tracking, event windows and rotation physics."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from config import Config
from tracking import Tracker
from detector import FixedEventDetector
from event_io import iter_windows, load_events
from rotation import score_fields, evaluate_slice as evaluate
from switch_frontend import build_maps, features, propose
from joint_score import score_candidate


def candidate(state='joint_candidate', box=(10, 10, 14, 14)):
    return {'box': box, 'state': state, 'joint_pass': state=='joint_candidate',
            'observable': True}


class UnifiedEvidenceTests(unittest.TestCase):
    def test_joint_weights_are_relative_ratios(self):
        f={'repetition_score':.7,'spatial_score':.3}
        r={'rotation_q':.4,'observable':True}
        a=score_candidate(f,r,Config(rotation_weight=.4,repetition_weight=.5,structure_weight=.1))
        b=score_candidate(f,r,Config(rotation_weight=40,repetition_weight=50,structure_weight=10))
        self.assertAlmostEqual(a['joint_score'],b['joint_score'])
        self.assertEqual(a['joint_pass'],b['joint_pass'])

    def test_switches_and_time_coverage_share_one_map(self):
        events=np.array([[3,4,.001,1],[3,4,.006,-1],[3,4,.015,1],[3,4,.025,-1],
                         [8,8,.002,1],[8,8,.002,-1]],dtype=float)
        cfg=Config()
        maps=build_maps(events,0,.033,16,16,cfg)
        self.assertEqual(maps.flips[4,3],3)
        self.assertEqual(maps.flips[8,8],0)
        f=features(maps,(0,0,16,16),cfg)
        self.assertEqual(f['occupied_bins'],3)
        self.assertAlmostEqual(maps.score[4,3],3/5*3/8,places=6)
        # The incoming packet order does not alter the per-pixel time ordering.
        reversed_maps=build_maps(events[::-1],0,.033,16,16,cfg)
        np.testing.assert_array_equal(maps.flips,reversed_maps.flips)

    def test_hot_pixel_saturates_and_is_not_detected(self):
        cfg=Config(); detector=FixedEventDetector(32,32,cfg)
        for window in range(6):
            start=window*.033
            times=np.linspace(start+.00001,start+.03299,400)
            events=np.column_stack([np.full(400,16),np.full(400,16),times,np.where(np.arange(400)%2,1,-1)])
            maps=build_maps(events,start,start+.033,32,32,cfg)
            self.assertLess(maps.score.max(),1)
            self.assertFalse(detector.process(events,start,start+.033)[0])

    def test_unobservable_rotation_is_unknown_without_moving_threshold(self):
        cfg=Config(); f={'repetition_score':1.,'spatial_score':1.}
        missing=score_candidate(f,{'rotation_q':.3,'observable':False},cfg)
        measured=score_candidate(f,{'rotation_q':.3,'observable':True},cfg)
        self.assertGreater(measured['joint_score'],missing['joint_score'])
        self.assertEqual(missing['joint_threshold'],measured['joint_threshold'])

    def test_rotation_contributes_to_joint_score(self):
        cfg=Config(joint_threshold=.525); f={'repetition_score':.25,'spatial_score':.8}
        positive=score_candidate(f,{'rotation_q':.9,'observable':True},cfg)
        missing=score_candidate(f,{'rotation_q':0.,'observable':False},cfg)
        self.assertTrue(positive['joint_pass'])
        self.assertFalse(missing['joint_pass'])

    def test_budget_is_not_measured_rotation(self):
        cfg=Config(); f={'repetition_score':1.,'spatial_score':1.}
        result=score_candidate(f,{'rotation_q':0.,'observable':False},cfg,deferred=True)
        self.assertFalse(result['joint_pass'])
        self.assertEqual(result['state'],'budget_deferred')

class TrackingTests(unittest.TestCase):
    def test_current_pass_is_published_immediately(self):
        tracker=Tracker(Config())
        output=tracker.update([candidate()],1)
        self.assertEqual(len(output),1)
        self.assertFalse(tracker.update([],2))
        self.assertFalse(tracker.update([candidate('joint_reject')],3))
        self.assertFalse(tracker.update([candidate('budget_deferred')],4))

    def test_association_preserves_identity(self):
        tracker=Tracker(Config())
        first=tracker.update([candidate()],1)[0]
        second=tracker.update([candidate(box=(11,10,15,14))],2)[0]
        self.assertEqual(first['track_id'],second['track_id'])
        self.assertEqual(second['box'],(11,10,15,14))

    def test_rejected_candidates_are_never_published(self):
        for state in ('budget_deferred', 'joint_reject'):
            tracker = Tracker(Config())
            for i in range(4):
                self.assertEqual(tracker.update([candidate(state)], i), [])

    def test_expired_identity_and_one_to_one(self):
        tracker = Tracker(Config())
        tracker.update([candidate()], 1)
        first = tracker.update([candidate()], 2)[0]['track_id']
        two=[candidate(), candidate()]
        tracker.update(two, 3)
        outputs = tracker.update([candidate(),candidate()], 4)
        self.assertEqual(len({o['track_id'] for o in outputs}), 2)
        tracker.update([candidate()],10)
        later = tracker.update([candidate()], 11)[0]['track_id']
        self.assertNotEqual(first, later)


class RotationTests(unittest.TestCase):
    def test_rotating_time_surface_through_ioc(self):
        y, x = np.mgrid[0:17, 0:17]
        timestamp = .024+(np.arctan2(y-8, x-8)+np.pi)/1000
        positive = np.column_stack([x.ravel(), y.ravel(), timestamp.ravel(), np.ones(x.size)])
        negative = positive.copy()
        negative[:, 3] = -1
        negative[:, 2] += .00001
        score = evaluate(np.concatenate([positive, negative]), (0, 0, 17, 17), .033, Config())
        self.assertTrue(score['observable'])
        self.assertGreaterEqual(score['rotation_q'], .45)

    def test_rigid_rotation_vs_translation_and_shear(self):
        cfg = Config()
        y, x = np.mgrid[-16:17, -16:17].astype(float)
        valid = np.ones(x.shape, bool)
        rotation = (-1000*y, 1000*x, valid)
        translation = (np.full(x.shape, 500.), np.full(x.shape, 200.), valid)
        shear = (1000*y, np.zeros_like(x), valid)
        scores=[]
        for field in (rotation, translation, shear):
            score = score_fields(field, field, cfg)
            self.assertTrue(score['observable'])
            scores.append(score['rotation_q'])
        self.assertGreaterEqual(scores[0],.45)
        self.assertLess(scores[1],.22)
        self.assertLess(scores[2],scores[0]/2)

    def test_shared_fit_accepts_both_polarity_conventions(self):
        cfg = Config()
        y, x = np.mgrid[-16:17, -16:17].astype(float)
        valid = np.ones(x.shape, bool)
        field = (-1000*y, 1000*x, valid)
        opposite = (1000*y, -1000*x, valid)
        same = score_fields(field, field, cfg)
        reversed_polarity = score_fields(field, opposite, cfg)
        self.assertGreater(same['rotation_q'], .95)
        self.assertGreater(reversed_polarity['rotation_q'], .95)
        self.assertEqual(same['rotation_polarity_sign'], 1)
        self.assertEqual(reversed_polarity['rotation_polarity_sign'], -1)
        self.assertGreater(same['rotation_blocks'], 2)

    def test_sparse_field_is_unknown_not_negative(self):
        valid = np.zeros((9, 9), bool)
        valid[4, 4] = True
        field = (np.zeros((9, 9)), np.zeros((9, 9)), valid)
        self.assertFalse(score_fields(field, field, Config())['observable'])


class IntegrationTests(unittest.TestCase):
    def test_gt_matching_counts_duplicates_and_small_boxes(self):
        from evaluate_gt import evaluate
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'frame_33333.txt').write_text('0 0.5 0.5 0.8 0.8\n')
            (root/'frame_66666.txt').write_text('')
            (root/'pred.csv').write_text('window,bbox_x0,bbox_y0,bbox_x1,bbox_y1\n1,40,40,50,50\n1,40,40,50,50\n2,40,40,50,50\n')
            result=evaluate(root/'pred.csv',root,2,100,100)
            self.assertEqual(result['ioa_0.5']['tp'],1)
            self.assertEqual(result['ioa_0.5']['fp'],2)
            self.assertEqual(result['iou_0.3']['tp'],0)
            self.assertEqual(result['iou_0.3']['fn'],1)

    def test_sparse_events_end_to_end(self):
        cfg = Config(joint_threshold=.20)
        detector = FixedEventDetector(64, 48, cfg)
        for window in range(4):
            start = window*0.033
            events = np.array([[20+j%4, 20+j//4, start+t, 1 if k%2 else -1]
                               for k,t in enumerate(np.arange(1,33)*.001) for j in range(8)])
            output, diagnostics, timing = detector.process(events, start, start+.033)
            self.assertTrue(diagnostics)
            self.assertLessEqual(timing['rotation_calls'], cfg.rotation_max_calls)
            self.assertTrue(output)
        self.assertEqual(detector.process(np.empty((0, 4)), .132, .165)[0], [])

    def test_persistent_bipolar_speckles_do_not_create_identity(self):
        detector = FixedEventDetector(64, 48, Config())
        for window in range(6):
            start = window*.033
            events = np.array([[20+j%2,20+(j//2)%2,start+t,1 if j%2 else -1]
                               for t in (.001,.012,.021) for j in range(16)])
            outputs, candidates, _ = detector.process(events,start,start+.033)
            self.assertFalse(outputs)
            self.assertTrue(all(c['repeat_pixels']==0 for c in candidates))

    def test_steady_event_rate_is_not_rejected(self):
        cfg=Config()
        events=np.array([[20+j%4,20+j//4,(k+.5)*.033/8,1 if k%2 else -1]
                         for k in range(8) for j in range(8)])
        maps=build_maps(events,0,.033,64,48,cfg)
        f=features(maps,(18,18,26,24),cfg)
        self.assertEqual(f['repeat_pixels'],8)
        self.assertGreater(f['repetition_score'],.4)

    def test_cross_file_windows_and_reset_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a.npz', Path(temp)/'b.npz'
            np.savez(a, events=np.array([[1, 1, 0., 1], [2, 2, .01, 0]]))
            np.savez(b, events=np.array([[3, 3, .02, 1], [4, 4, .04, 0]]))
            windows = list(iter_windows([a, b], 's', 10, 10, 33))
            self.assertEqual([len(w[1]) for w in windows], [3, 1])
            self.assertEqual(windows[0][1][1, 3], -1)
            with self.assertRaises(ValueError):
                list(iter_windows([b, a], 's', 10, 10, 33))

    def test_h5_and_invalid_coordinates(self):
        import h5py
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'a.h5'
            with h5py.File(path, 'w') as f:
                group = f.create_group('events')
                for key, values in {'x': [1, 2], 'y': [1, 2], 't': [1000, 2000], 'p': [0, 1]}.items():
                    group[key] = values
            events = load_events(path, 'us', 10, 10)
            np.testing.assert_allclose(events[:, 2], [.001, .002])
            with self.assertRaises(ValueError):
                load_events(path, 'us', 1, 1)

    def test_partial_overlap_trim_and_independent_files(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a.npz', Path(temp)/'b.npz'
            np.savez(a, events=np.array([[1, 1, 0., 1], [2, 2, .02, 0]]))
            np.savez(b, events=np.array([[3, 3, .01, 1], [4, 4, .04, 0]]))
            with self.assertWarns(UserWarning):
                windows = list(iter_windows([a, b], 's', 10, 10, 33, overlap_policy='trim'))
            self.assertEqual(sum(len(w[1]) for w in windows), 3)
            independent = list(iter_windows([a, a], 's', 10, 10, 33, file_mode=True))
            self.assertEqual(len(independent), 2)
            self.assertGreaterEqual(independent[1][1][0, 2], independent[0][3])


if __name__ == '__main__':
    unittest.main()

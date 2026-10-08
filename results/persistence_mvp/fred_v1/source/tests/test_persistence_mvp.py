import unittest
from dataclasses import replace
from persistence.config import Config
from persistence.tracker import Tracker
from persistence.evaluation import match_boxes,evaluate,align_frames
from persistence.replay import replay
from test_persistence_primitives import obs


def cfg(**kw):
    return replace(Config(enabled=True,cold_start='fixed',sigma_position=1.,sigma_direction=1.,
                          sigma_acceleration=100.,threshold=.3,radius=10),**kw)


class TrackerTests(unittest.TestCase):
    def test_short_miss_and_no_hallucination(self):
        tracker=Tracker(cfg())
        for t in range(3):
            rows,_,_=tracker.step(t,[obs(t,t)])
        track_id=rows[0]['track_id']
        rows,changes,_=tracker.step(3,[])
        self.assertEqual(rows,[]);self.assertEqual(changes[0]['state'],'Lost')
        rows,_,_=tracker.step(4,[obs(4,4)])
        self.assertEqual(rows[0]['track_id'],track_id)
        self.assertEqual(rows[0]['history'][-1],(4.,0.))

    def test_deletion_strict_exceeds(self):
        tracker=Tracker(cfg())
        tracker.step(0,[obs(0)])
        tracker.step(1,[]);tracker.step(2,[])
        self.assertEqual(len(tracker.tracks),1)
        _,changes,_=tracker.step(3,[])
        self.assertEqual(changes[0]['state'],'Deleted')
        self.assertEqual(len(tracker.tracks),0)
        rows,_,_=tracker.step(4,[obs(4)])
        self.assertEqual(rows[0]['track_id'],2)

    def test_one_to_one_and_gate(self):
        tracker=Tracker(cfg())
        tracker.step(0,[obs(0)])
        rows,_,_=tracker.step(1,[obs(1,1),obs(1,2),obs(1,100)])
        self.assertEqual(len({r['track_id'] for r in rows}),3)
        self.assertEqual(rows[0]['track_id'],1)

    def test_reordering(self):
        a,b=Tracker(cfg()),Tracker(cfg())
        x=[obs(0,20),obs(0,0)]
        ra,_,_=a.step(0,x);rb,_,_=b.step(0,x[::-1])
        self.assertEqual({r['bbox']:r['track_id'] for r in ra},{r['bbox']:r['track_id'] for r in rb})

    def test_history_bounded_and_rejected_candidates_retained(self):
        tracker=Tracker(cfg(threshold=1.))
        for t in range(30):tracker.step(t,[obs(t)])
        track=tracker.tracks[1]
        self.assertEqual(len(track.detections),8);self.assertEqual(len(track.hits),8)
        self.assertEqual(track.hit_count,30)

    def test_duplicate_window_rejected(self):
        tracker=Tracker(cfg());tracker.step(0,[])
        with self.assertRaises(ValueError):tracker.step(0,[])

    def test_t_only_warmup_and_masks(self):
        tracker=Tracker(cfg(use_neighborhood=False))
        rows,_,_=tracker.step(0,[obs(0)])
        self.assertFalse(rows[0]['accepted']);self.assertIsNone(rows[0]['persistence_score'])
        tracker.step(1,[obs(1)])
        rows,_,_=tracker.step(2,[obs(2)])
        self.assertTrue(rows[0]['accepted']);self.assertIsNone(rows[0]['direction_change'])

    def test_disabled_passes_every_current_candidate(self):
        tracker=Tracker(Config())
        rows,_,_=tracker.step(0,[obs(0),obs(0,10)])
        self.assertTrue(all(r['accepted'] for r in rows))

    def test_random_outside_gate_new_tracks(self):
        tracker=Tracker(cfg())
        ids=[]
        for t,x in enumerate([0,100,400,-200,700]):
            rows,_,_=tracker.step(t,[obs(t,x)]);ids.append(rows[0]['track_id'])
        self.assertEqual(len(set(ids)),5)


class EvaluationTests(unittest.TestCase):
    def test_duplicates_count_false_positive(self):
        box=(0,0,10,10)
        result,_=evaluate([dict(timestamp=1,window_id=0,lag_sec=0,gt=[dict(bbox=box)])],{0:[box,box]},'iou50')
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,0))

    def test_maximum_matching(self):
        gt=[(0,0,10,10),(5,0,15,10)]
        # First candidate can match both; second only the first.
        matches=match_boxes([(6,2,8,4),(1,2,3,4)],gt,'center')
        self.assertEqual(len(matches),2)

    def test_no_future_or_stale_alignment(self):
        windows=[dict(frame=0,end_sec=.03),dict(frame=1,end_sec=.06)]
        aligned=align_frames(windows,[10000,33333,66666,120000],{})
        self.assertEqual([r['window_id'] for r in aligned],[0,1])

    def test_center_not_box_metric(self):
        pred=[(4,4,6,6)];gt=[(0,0,10,10)]
        self.assertEqual(len(match_boxes(pred,gt,'center')),1)
        self.assertEqual(len(match_boxes(pred,gt,'iou50')),0)

    def test_no_predictions_precision_undefined(self):
        result,_=evaluate([dict(timestamp=0,window_id=0,lag_sec=0,gt=[dict(bbox=(0,0,1,1))])],{},'iou50')
        self.assertIsNone(result['precision']);self.assertEqual(result['recall'],0)


if __name__=='__main__':unittest.main()

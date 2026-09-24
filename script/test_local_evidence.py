import unittest
from unittest.mock import patch
import numpy as np
from config import Config
from switch_frontend import SwitchMaps, features
from joint_score import score_candidate


class LocalEvidenceTests(unittest.TestCase):
    def maps(self):
        shape=(20,20)
        flips=np.zeros(shape,np.int32);flips[8:12,8:12]=4
        bits=np.zeros(shape,np.uint16);bits[8:12,8:12]=255
        activity=np.ones(shape,np.int32);activity[8:12,8:12]=20
        return SwitchMaps(flips,bits,activity,np.zeros(shape,np.float32))

    def test_same_pixels_beat_disjoint_time_coverage(self):
        cfg=Config(); a=self.maps();b=self.maps()
        b.time_bits[8:12,8:12]=(1 << (np.arange(16).reshape(4,4)%8)).astype(np.uint16)
        fa,fb=[features(m,(6,6,14,14),cfg) for m in (a,b)]
        self.assertEqual(fa['repetition_base'],fb['repetition_base'])
        self.assertEqual(fa['persistent_switch_support'],1)
        self.assertEqual(fb['persistent_switch_support'],0)
        self.assertGreater(fa['repetition_score'],fb['repetition_score'])

    def test_background_activity_lowers_score_without_changing_roi(self):
        a=self.maps();b=self.maps()
        b.activity[:6,:]=100;b.activity[14:,:]=100;b.activity[:,:6]=100;b.activity[:,14:]=100
        fa,fb=[features(m,(6,6,14,14),Config()) for m in (a,b)]
        self.assertEqual(fa['foreground_density'],fb['foreground_density'])
        self.assertEqual(fa['repetition_base'],fb['repetition_base'])
        self.assertGreater(fa['background_contrast'],fb['background_contrast'])
        self.assertGreater(fa['repetition_score'],fb['repetition_score'])

    def test_border_and_empty_maps_are_finite(self):
        m=self.maps()
        for box in ((0,0,20,20),(0,0,3,3),(18,18,20,20)):
            f=features(m,box,Config())
            self.assertTrue(all(np.isfinite(v) for v in f.values()))
            self.assertGreaterEqual(f['background_contrast'],0)
            self.assertLessEqual(f['background_contrast'],1)
        self.assertEqual(features(m,(0,0,20,20),Config())['background_contrast'],1)

    def test_zero_penalties_equal_base_repetition(self):
        f=features(self.maps(),(6,6,14,14),Config(persistence_penalty=0,background_penalty=0))
        self.assertEqual(f['repetition_score'],f['repetition_base'])

    def test_observability_changes_rotation_evidence_not_threshold(self):
        cfg=Config();f={'repetition_score':.99,'spatial_score':.6}
        low=score_candidate(f,{'observable':False,'rotation_q':.3},cfg)
        high=score_candidate(f,{'observable':True,'rotation_q':.3},cfg)
        self.assertLess(low['joint_score'],high['joint_score'])
        self.assertEqual(low['joint_threshold'],high['joint_threshold'])

    def test_invalid_new_parameters_rejected(self):
        for kwargs in ({'persistence_penalty':1.1},{'background_penalty':-.1},
                       {'background_ring_px':0},{'background_ring_px':1.5}):
            with self.assertRaises(ValueError):Config(**kwargs).validate()

    def test_rotation_label_distinguishes_missing_from_measured_zero(self):
        from PIL import ImageDraw
        from visualization import render_frame
        texts=[]
        def capture(draw, xy, text, **kwargs):texts.append(text)
        with patch.object(ImageDraw.ImageDraw, 'text', capture):
            for n in (0,1):
                render_frame(np.empty((0,4)),[dict(box=(5,20,20,40),observable=bool(n),
                    rotation_q=0.,rotation_omega=12.,rotation_fit=.4,
                    rotation_rigidity=.5)],320,100)
        self.assertTrue(any('Rot=N/A' in t for t in texts))
        self.assertTrue(any('Rot=0.00 Om=12 F=0.40 G=0.50' in t for t in texts))


if __name__=='__main__':unittest.main()

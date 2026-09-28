"""Area-filter contracts: unchanged legacy path, exact boundary, no noise resurrection."""
import unittest
import tempfile
import csv
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy import ndimage
from evdetmav.saliency import remove_small_saliency_regions
from evdetmav.pipeline import process_window
from evdetmav.cli import build_parser
from evdetmav.clustering import initialize_candidates, refine_candidate
from evdetmav.models import Candidate

class SmallRegionTests(unittest.TestCase):
    def test_disabled_is_exact_identity(self):
        values=np.array([[0,25.5,51],[102,255,1]],np.float32)
        out,stats=remove_small_saliency_regions(values,50,0)
        self.assertIs(out,values)
        self.assertFalse(stats['enabled'])

    def test_exact_boundary_and_real_pixel_area(self):
        values=np.zeros((14,18),np.float32)
        values[1:4,1:4]=102  # 9 pixels: kept
        values[7:9,1:5]=255  # 8 pixels: removed
        values[2,9]=51;values[3,10]=51;values[4,11]=51  # box area 9, actual area 3
        before=values.copy()
        out,stats=remove_small_saliency_regions(values,50,9)
        self.assertEqual(np.count_nonzero(out),9)
        np.testing.assert_array_equal(out[1:4,1:4],before[1:4,1:4])
        np.testing.assert_array_equal(values,before)
        self.assertEqual(stats['components_removed'],2)
        self.assertEqual(stats['foreground_pixels_removed'],11)

    def test_eight_connectivity(self):
        values=np.eye(3,dtype=np.float32)*51
        out,_=remove_small_saliency_regions(values,50,3)
        self.assertEqual(np.count_nonzero(out),3)

    def test_empty_and_background(self):
        for values in [np.zeros((5,5),np.float32),np.ones((5,5),np.float32)*25.5]:
            out,stats=remove_small_saliency_regions(values,50,9)
            self.assertFalse(out.any());self.assertEqual(stats['components_before'],0)

    def test_invalid_parameters(self):
        for threshold,area in [(50,-1),(0,9),(float('nan'),9)]:
            with self.assertRaises(ValueError):remove_small_saliency_regions(np.zeros((2,2)),threshold,area)

    def test_dilation_merge_and_fallback_cannot_recover_removed_pixels(self):
        args=build_parser().parse_args(['--input','dummy','--saliency-min-area','9','--init-dilate-px','2'])
        values=np.zeros((30,30),np.float32);values[4:7,4:7]=102;values[8,8]=255
        filtered,_=remove_small_saliency_regions(values,50,9)
        candidates=initialize_candidates(filtered,args)
        self.assertTrue(candidates)
        for candidate in candidates:
            refined=refine_candidate(filtered,candidate,args)
            if refined is not None:
                self.assertFalse(refined.mask[8-refined.offset_y,8-refined.offset_x])
        self.assertEqual(filtered[8,8],0)

    def test_pipeline_filters_before_periodicity(self):
        args=build_parser().parse_args(['--input','dummy','--saliency-min-area','9','--init-dilate-px','2'])
        values=np.zeros((12,12),np.float32);values[6,6]=255
        empty=np.array([],dtype=np.int64)
        with patch('evdetmav.pipeline.build_density_saliency',return_value=values),patch('evdetmav.pipeline.evaluate_periodicity') as periodicity:
            result=process_window(empty,empty,empty,empty,0,.03,12,12,'fixture',0,args)
        periodicity.assert_not_called()
        self.assertFalse(result.saliency_u8.any());self.assertEqual(result.detections,[])
        self.assertEqual(result.saliency_filter_stats['foreground_pixels_removed'],1)
        self.assertEqual(result.raw_saliency_u8[6,6],255)

    def test_cropped_fragment_cannot_pass_fine_fallback(self):
        args=build_parser().parse_args(['--input','dummy','--saliency-min-area','9'])
        values=np.zeros((20,20),np.float32);values[3:6,3:6]=102
        candidate=Candidate(box=(5,5,12,12),area_px=1,saliency_score=102)
        self.assertIsNone(refine_candidate(values,candidate,args))

    def test_every_surviving_region_meets_area_limit(self):
        rng=np.random.default_rng(42)
        values=rng.choice([0,25.5,51,102,255],size=(40,50)).astype(np.float32)
        out,_=remove_small_saliency_regions(values,50,9)
        labels,count=ndimage.label(out>0,structure=np.ones((3,3)))
        areas=np.bincount(labels.ravel())
        self.assertTrue(np.all(areas[1:]>=9))
        np.testing.assert_array_equal(out[out>0],values[out>0])

    def test_h5_runner_records_area_filter(self):
        import h5py
        from persistence.runner import run
        from persistence.config import Config
        root=Path(__file__).resolve().parents[1]
        values=np.array([(x,y,i*3000+pol,pol) for i in range(10) for x in [4,5] for y in [4,5] for pol in [0,1]],
            dtype=[('x','u2'),('y','u2'),('t','i8'),('p','i2')])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.h5';out=Path(directory)/'out'
            with h5py.File(path,'w') as hf:hf.create_dataset('events',data=values)
            run(Config.load(root/'configs/persistence_original.json'),['--input',str(path),'--out',str(out),
                '--width','16','--height','16','--time-unit','us','--window-ms','30','--step-ms','30',
                '--saliency-min-area','9','--intersection-radius','0','--tau-s','1','--progress-every','0'])
            with (out/'saliency_filter_windows.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1);self.assertEqual(int(rows[0]['components_removed']),1)
            self.assertEqual(int(rows[0]['foreground_pixels_removed']),4)
            self.assertEqual(json.loads((out/'source.json').read_text())['saliency_min_area_px'],9)

if __name__=='__main__':unittest.main()

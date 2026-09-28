"""Real spatial support must never be fabricated by repetition or dilation."""
import unittest
import tempfile
import csv
from pathlib import Path
import numpy as np
from evdetmav.cli import build_parser
from evdetmav.saliency import raw_pixel_support_mask, build_density_saliency
from evdetmav.pipeline import process_window

class RawSupportTests(unittest.TestCase):
    def test_repeated_single_coordinate_stays_one_pixel(self):
        x=np.full(1000,5,dtype=int);y=np.full(1000,5,dtype=int)
        keep,stats=raw_pixel_support_mask(x,y,12,12,3)
        self.assertFalse(keep.any())
        self.assertEqual(stats['raw_unique_pixels_before'],1)
        self.assertEqual(stats['raw_events_removed'],1000)

    def test_distinct_coordinates_not_polarity_or_frequency(self):
        x=np.array([2,2,3,3,8,8,9,10]);y=np.array([2,2,2,2,8,8,8,8])
        keep,stats=raw_pixel_support_mask(x,y,16,16,3)
        np.testing.assert_array_equal(keep,[False,False,False,False,True,True,True,True])
        self.assertEqual(stats['raw_unique_pixels_removed'],2)
        self.assertEqual(stats['raw_components_removed'],1)

    def test_diagonal_connectivity_and_boundary(self):
        keep,_=raw_pixel_support_mask(np.array([0,1,2]),np.array([0,1,2]),8,8,3)
        self.assertTrue(keep.all())

    def test_disabled_and_empty(self):
        empty=np.array([],dtype=int)
        keep,stats=raw_pixel_support_mask(empty,empty,4,4,0)
        self.assertIsNone(keep);self.assertFalse(stats['raw_support_enabled'])
        keep,stats=raw_pixel_support_mask(empty,empty,4,4,3)
        self.assertEqual(keep.size,0);self.assertEqual(stats['raw_components_before'],0)

    def test_negative_threshold(self):
        with self.assertRaises(ValueError):raw_pixel_support_mask(np.array([1]),np.array([1]),3,3,-1)

    def fixture(self):
        # 同一像素在十个时间片重复产生正负事件：旧空间容差将其扩成 3×3。
        x=np.full(20,5,dtype=int);y=np.full(20,5,dtype=int)
        t=np.repeat(np.arange(10)*.003,2)+np.tile([.0001,.0002],10)
        p=np.tile([1,-1],10)
        return x,y,t,p

    def test_single_pixel_is_removed_before_any_dilation(self):
        x,y,t,p=self.fixture()
        old=build_parser().parse_args(['--input','dummy'])
        before=build_density_saliency(x,y,t,p,0,.03,12,12,old)
        self.assertEqual(np.count_nonzero(before),9)
        for radius in [0,1,2,3]:
            args=build_parser().parse_args(['--input','dummy','--intersection-radius',str(radius),'--min-raw-component-pixels','3'])
            result=process_window(x,y,t,p,0,.03,12,12,'fixture',0,args)
            self.assertFalse(result.saliency_u8.any());self.assertEqual(result.detections,[])
            self.assertEqual(result.saliency_filter_stats['raw_unique_pixels_removed'],1)

    def test_sufficient_real_support_preserves_saliency_exactly(self):
        x,y,t,p=self.fixture()
        x=np.concatenate([x,x+1,x+2]);y=np.tile(y,3);t=np.tile(t,3);p=np.tile(p,3)
        args=build_parser().parse_args(['--input','dummy'])
        before=build_density_saliency(x,y,t,p,0,.03,12,12,args)
        args.min_raw_component_pixels=3
        after=build_density_saliency(x,y,t,p,0,.03,12,12,args)
        np.testing.assert_array_equal(after,before)

    def test_separate_single_points_cannot_combine_by_dilation(self):
        x=np.array([2,2,4,4]);y=np.array([3,3,3,3]);t=np.array([.001,.002,.001,.002]);p=np.array([1,-1,1,-1])
        args=build_parser().parse_args(['--input','dummy','--min-raw-component-pixels','3','--intersection-radius','3'])
        saliency=build_density_saliency(x,y,t,p,0,.03,12,12,args)
        self.assertFalse(saliency.any())

    def test_streaming_runner_logs_new_filter_with_old_filter_off(self):
        import h5py
        from persistence.runner import run
        from persistence.config import Config
        root=Path(__file__).resolve().parents[1]
        events=np.array([(5,5,i*3000+pol,pol) for i in range(10) for pol in [0,1]],
            dtype=[('x','u2'),('y','u2'),('t','i8'),('p','i2')])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.h5';out=Path(temp)/'result'
            with h5py.File(path,'w') as f:f.create_dataset('events',data=events)
            run(Config.load(root/'configs/persistence_original.json'),['--input',str(path),'--out',str(out),
                '--width','16','--height','16','--time-unit','us','--window-ms','30','--step-ms','30',
                '--min-raw-component-pixels','3','--saliency-min-area','0','--progress-every','0'])
            with (out/'saliency_filter_windows.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['raw_support_enabled'],'True')
            self.assertEqual(rows[0]['enabled'],'False')  # 旧的膨胀后面积过滤确实关闭。
            self.assertEqual(int(rows[0]['raw_unique_pixels_removed']),1)
            self.assertEqual(int(rows[0]['raw_events_removed']),20)

if __name__=='__main__':unittest.main()

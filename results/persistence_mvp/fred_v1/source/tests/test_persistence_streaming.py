import unittest
import tempfile
from pathlib import Path
import numpy as np
import h5py
from evdetmav.cli import build_parser,build_windows
from persistence.events import stream_windows
from persistence.config import Config
from dataclasses import replace
from unittest.mock import patch
from persistence.baseline_adapter import detect


class StreamingTests(unittest.TestCase):
    def test_matches_inclusive_cli_boundaries_and_empty_windows(self):
        values=np.array([(1,2,0,1),(2,3,30000,-1),(3,4,95000,1),(3,4,100000,1)],
                        dtype=[('x','u2'),('y','u2'),('t','i8'),('p','i2')])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.h5'
            with h5py.File(path,'w') as hf:hf.create_dataset('events',data=values,chunks=(2,))
            args=build_parser().parse_args(['--input',str(path),'--window-ms','30','--step-ms','30'])
            result=list(stream_windows(path,args));times=values['t']*1e-6
            expected=build_windows(times,args)
            self.assertEqual(len(result),len(expected))
            for (wid,s,e,events),(ew,es,ee) in zip(result,expected):
                np.testing.assert_array_equal(events,values[(times>=s)&(times<=e)])
            self.assertEqual(len(result[2][3]),0)
            self.assertIn(30000,result[0][3]['t']);self.assertIn(30000,result[1][3]['t'])

    def test_true_periodicity_off_does_not_call_original_pipeline(self):
        args=build_parser().parse_args(['--input','dummy.h5','--min-window-events','1'])
        values=np.array([(1,2,0,1)],dtype=[('x','u2'),('y','u2'),('t','i8'),('p','i2')])
        with patch('persistence.baseline_adapter.process_window',side_effect=AssertionError('must not run')):
            rows=detect(values,0,.03,10,10,'dummy',0,args,use_periodicity=False)
            self.assertEqual(rows,[])

    def test_debug_panel_preserves_entire_sensor_image(self):
        from PIL import Image
        from persistence.visualization import debug_panel
        image=Image.new('RGB',(1280,784),(12,34,56))
        result=debug_panel(image,[])
        self.assertEqual(result.size,(1280,928))
        self.assertEqual(result.crop((0,0,1280,784)).tobytes(),image.tobytes())

    def test_negative_config_values(self):
        base=Config(enabled=True,cold_start='fixed',sigma_position=1,sigma_direction=1,sigma_acceleration=1,threshold=.4)
        for update in [dict(radius=-1),dict(temporal_decay=0),dict(lambda_position=-1),dict(epsilon=0),dict(alpha=2)]:
            with self.assertRaises(ValueError):replace(base,**update).validate()
        replace(base,use_trajectory=False,sigma_position=None,sigma_direction=None,sigma_acceleration=None).validate()


if __name__=='__main__':unittest.main()

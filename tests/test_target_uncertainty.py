import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy is optional for summary CI')
class WilsonTests(unittest.TestCase):
    def test_extreme_and_symmetric_counts(self):
        from target_uncertainty import wilson
        low,high=wilson(0,92)
        self.assertEqual(low,0);self.assertGreater(high,0)
        a,b=wilson(92,92);self.assertEqual(b,1);self.assertAlmostEqual(a,1-high)
        a,b=wilson(46,92);self.assertAlmostEqual(a,1-b)
        with self.assertRaises(ValueError):wilson(0,0)
if __name__=='__main__':unittest.main()

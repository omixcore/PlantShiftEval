import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
AVAILABLE=all(importlib.util.find_spec(n) for n in ('numpy','scipy'))

@unittest.skipUnless(AVAILABLE,'Statistical dependencies are optional for summary CI')
class SeedStatisticsTests(unittest.TestCase):
    def test_pairs_use_intersection_and_difference_direction(self):
        from summarize_corrected_study import paired_effect
        r=paired_effect({0:.8,1:.9,2:.7,9:0},{0:.6,1:.8,2:.6})
        self.assertEqual(r['n'],3)
        self.assertEqual(r['seeds'],[0,1,2])
        self.assertAlmostEqual(r['mean_difference'],.4/3)
        self.assertLess(r['ci95_student_t'][0],r['mean_difference'])
        self.assertGreater(r['ci95_student_t'][1],r['mean_difference'])
        self.assertEqual(paired_effect({0:.8},{1:.6})['status'],'insufficient_seeds')
    def test_holm_adjustment_is_monotone_and_bounded(self):
        from summarize_corrected_study import holm
        rows=[{'p_unadjusted':.01},{'p_unadjusted':.03},{'p_unadjusted':.9}]
        holm(rows)
        self.assertEqual([r['p_holm'] for r in rows],[.03,.06,.9])

if __name__=='__main__':unittest.main()

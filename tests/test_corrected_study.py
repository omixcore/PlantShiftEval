import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_corrected_study import matrix

class MatrixTests(unittest.TestCase):
    def test_prespecified_matrix_has_all_paired_seeds_and_unique_cells(self):
        jobs=matrix()
        self.assertEqual(len(jobs),65)
        self.assertEqual(len({j['run_id'] for j in jobs}),65)
        for model in ('mtdnet','smallcnn'):
            for ep in (6,12):
                for lr in ('1e-3','2e-3'):
                    self.assertEqual({j['seed'] for j in jobs if (j['model'],j['epochs'],j['lr'])==(model,ep,lr)},set(range(5)))
        for model in ('mtdnet_pt','resnet18','mtd_concat','mtd_single','mtd_appearance'):
            self.assertEqual(sum(j['model']==model for j in jobs),5)

if __name__=='__main__':unittest.main()

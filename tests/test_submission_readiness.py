import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from target_bootstrap import class_metrics, bootstrap_indices, mcnemar_exact, holm
from audit_manuscript import parse_reference_lines, parse_citations, title_from_ref, doi_from_ref

class TargetBootstrapTests(unittest.TestCase):
    def test_supported_macro_f1_and_accuracy(self):
        y=np.array([0,0,1,1]); pred=np.array([0,1,1,1])
        m=class_metrics(y,pred)
        self.assertAlmostEqual(m["accuracy"],.75)
        self.assertEqual(m["n_supported_classes"],2)
        self.assertAlmostEqual(m["macro_f1_supported"],(2/3+0.8)/2)

    def test_stratified_bootstrap_preserves_class_support(self):
        y=np.array([0,0,0,1,1,2])
        samples=bootstrap_indices(y,reps=20,seed=7,stratified=True)
        self.assertEqual(len(samples),20)
        for ix in samples:
            vals,cnts=np.unique(y[ix],return_counts=True)
            self.assertEqual(dict(zip(vals.tolist(),cnts.tolist())),{0:3,1:2,2:1})

    def test_mcnemar_exact_and_holm(self):
        self.assertEqual(mcnemar_exact([1,1,0],[0,1,1])["mcnemar_exact_p"],1.0)
        self.assertAlmostEqual(mcnemar_exact([1,1,1,1,1],[0,0,0,0,0])["mcnemar_exact_p"],.0625)
        self.assertEqual(holm([.01,.03,.9]),[.03,.06,.9])

class ManuscriptAuditTests(unittest.TestCase):
    def test_reference_numbering_and_citation_ranges(self):
        text="\\n".join(["## Intro","Claim [1, 2] and [3–4].","## References","1. A. Title.","","2. B. Title.","","3. C. Title.","","4. D. Title."])
        refs=parse_reference_lines(text)
        self.assertEqual(sorted(refs),[1,2,3,4])
        cited,invalid=parse_citations(text,4)
        self.assertEqual(cited,[1,2,3,4]);self.assertEqual(invalid,[])

    def test_title_and_doi_extraction(self):
        ref="1. Mohanty SP, Hughes DP, Salathé M. Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*. 2016;7:1419. https://doi.org/10.3389/fpls.2016.01419"
        title,_=title_from_ref(ref)
        self.assertEqual(title,"Using Deep Learning for Image-Based Plant Disease Detection")
        self.assertEqual(doi_from_ref(ref),"10.3389/fpls.2016.01419")

if __name__=="__main__": unittest.main()

import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
AVAILABLE=all(importlib.util.find_spec(n) for n in ('numpy','pyarrow','PIL'))

@unittest.skipUnless(AVAILABLE,'Data-audit dependencies are optional for summary CI')
class DataAuditTests(unittest.TestCase):
    def test_transitive_identity_and_leaf_components(self):
        from prepare_corrected_data import component_ids
        def row(name,raw,pixel,leaf):return dict(name=name,sha256=raw,pixel_sha256=pixel,leaf_groups=leaf)
        rows=[row('a','x','p',['leaf1']),row('b','y','q',['leaf1']),row('c','y','r',[]),row('d','z','s',[])]
        ids=component_ids(rows)
        self.assertEqual(ids[0],ids[2]);self.assertNotEqual(ids[2],ids[3])
    def test_group_split_is_disjoint_covers_rows_and_preserves_classes(self):
        from prepare_corrected_data import group_split
        labels=[0,0,0,0,1,1,1,1];groups=['a','a','b','b','c','c','d','d']
        train,val=group_split(labels,groups)
        self.assertEqual(set(train)|set(val),set(range(8)))
        self.assertFalse({groups[i] for i in train}&{groups[i] for i in val})
        self.assertEqual({labels[i] for i in val},{0,1})
        with self.assertRaises(ValueError):group_split([0,0],['a','a'])

if __name__=='__main__':unittest.main()

"""Optional real PyTorch smoke tests; skipped in dependency-free summary CI."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
AVAILABLE=all(importlib.util.find_spec(n) for n in ('torch','torchvision','pyarrow','scipy','sklearn'))

@unittest.skipUnless(AVAILABLE,'PyTorch runtime dependencies are optional for summary CI')
class TrainingRuntimeTests(unittest.TestCase):
    def test_validation_selects_and_test_is_evaluated_once(self):
        import argparse,numpy as np,torch
        import train
        from data import Split
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.fc=torch.nn.Linear(3,3)
            def forward(self,x):return {'disease':self.fc(x.mean((2,3))),'species':None}
        def data(n):return Split(np.random.RandomState(n).randint(0,255,(n,32,32,3),dtype=np.uint8),np.arange(n)%3,np.zeros(n,dtype=np.int64))
        tr,va,te=data(12),data(6),data(9)
        seen=[];original=train.evaluate
        def evaluate(model,loader,device):
            seen.append(loader.dataset.x)
            return original(model,loader,device)
        with tempfile.TemporaryDirectory() as out:
            args=argparse.Namespace(seed=0,threads=1,model='tiny',bg_aug=False,
                crop_balance=False,size=16,bs=4,epochs=2,lr=.001,wd=.02,
                aux_w=.3,out=out,bg_p=.7,val_frac=.1)
            with patch.object(train,'build_model',return_value=Tiny()),patch.object(train,'evaluate',side_effect=evaluate):
                result=train.train_one(args,{'n_classes':3,'n_species':1},tr,te,va)
            self.assertEqual(sum(x is va.images for x in seen),2)
            self.assertEqual(sum(x is te.images for x in seen),1)
            self.assertEqual(result['test_evaluations'],1)
            self.assertEqual(set(result['history'][0])&{'test_acc','test_macro_f1'},set())
            self.assertTrue((Path(out)/'model.pt').exists())
            self.assertTrue((Path(out)/'logits.pt').exists())

if __name__=='__main__':unittest.main()

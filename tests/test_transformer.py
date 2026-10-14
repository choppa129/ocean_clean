import importlib.util
import unittest
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('torch'),'Install optional deep dependencies')
class TransformerTests(unittest.TestCase):
    def test_hidden_labels_are_not_inputs(self):
        from ocean_clean.transformer import MaskedWindows
        a=np.ones((300,2,4),dtype='float32');a[:,0,0]=np.arange(300)
        mean=np.zeros((2,4),dtype='float32');std=np.ones((2,4),dtype='float32')
        ds=MaskedWindows([a],mean,std,samples=4)
        values,observed,truth,mask=ds[0]
        self.assertTrue((values[mask,0,0]==0).all())
        self.assertTrue((observed[mask,0,0]==0).all())
        self.assertTrue((observed[mask,1,0]==1).all())
        year,start,offset,gap=ds.examples[0]
        a[start+offset:start+offset+gap,0,0]+=100
        values2,observed2,truth2,_=ds[0]
        self.assertTrue((values==values2).all())
        self.assertTrue((observed==observed2).all())
        self.assertFalse((truth[mask]==truth2[mask]).all())

    def test_forward_backward_with_missing_neighbor(self):
        import torch
        from ocean_clean.transformer import OceanTransformer
        torch.set_num_threads(2)
        model=OceanTransformer(4)
        x=torch.randn(2,128,2,4);mask=torch.ones_like(x)
        x[:,:,1]=0;mask[:,:,1]=0
        out=model(x,mask)
        self.assertEqual(tuple(out.shape),(2,128))
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertIsNotNone(model.input.weight.grad)

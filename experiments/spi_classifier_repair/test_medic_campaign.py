import unittest
import numpy as np
import torch
from experiments.spi_classifier_repair.medic_campaign import TokenHead,fit_token,token_predict,folds

class CampaignTests(unittest.TestCase):
    def test_padding_cannot_change_logits(self):
        torch.manual_seed(17);torch.set_num_threads(1)
        x=torch.randn(2,6,8);mask=torch.tensor([[1,1,1,0,0,0]]*2,dtype=torch.bool)
        changed=x.clone();changed[:,3:]=1000
        for kind in ('mean_mlp','gated_abmil','single_query_cross_attention'):
            net=TokenHead(8,kind).eval()
            with torch.no_grad():self.assertTrue(torch.allclose(net(x,mask),net(changed,mask),atol=1e-5))
    def test_real_optimizer_step_and_export_parity(self):
        torch.set_num_threads(1);x=np.random.default_rng(17).normal(size=(12,4,8)).astype('float32')
        mask=np.ones((12,4),bool);y=np.array([0,1,2,3]*3)
        net,h,e=fit_token(x,mask,y,'gated_abmil',.001,epochs=2,device='cpu')
        p=token_predict(net,x,mask);other=TokenHead(8,'gated_abmil');other.load_state_dict(net.state_dict())
        self.assertTrue(np.isfinite(p).all());self.assertTrue(np.allclose(p.sum(1),1));self.assertEqual(len(h),2)
        self.assertTrue(np.allclose(p,token_predict(other,x,mask),atol=1e-4))
    def test_missing_class_fails(self):
        with self.assertRaises(ValueError):folds(np.array([0,1]*6),np.arange(12))

if __name__=='__main__':unittest.main()

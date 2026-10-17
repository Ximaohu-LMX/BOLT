import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.lb_order import make_model, ForbiddenUpperTable
from src.CLIQUE_bound import CLIQUEestimator_bound
from src.CLIQUE_bound_mono import CLIQUEestimator_bound_mono
from src.model import msle_loss
from src.util import hinge_monotonicity_loss

CFG=dict(N=5,PT=20,seed=0,ch_es=4,rnn_hs=8,pred_hs=4,n_pred_layer=2,
 learning_rate=.001,l2=.000001,n_epoch=1,batch_size=8,patience=2,dynamicPT=True,
 max_entry_ratio=4.8,batch_norm=False,lambda_mono=5.,mono_tolerance=.1)
CHARS={'<pad>':0,'a':1,'b':2,'%':3,'_':4}

def model():
 m=make_model(CFG,CHARS);m.n_db=100
 m.substr_library={'exact':{'aba':4},'prefix':{'aba':8},'suffix':{'aba':7},'infix':{'aba':12}}
 return m

class LBOrderTests(unittest.TestCase):
 def test_no_table_build_or_head_feature(self):
  m=model()
  with self.assertRaises(RuntimeError):m.build_count('ANY')
  with self.assertRaises(RuntimeError):_='aba' in m.hashtable
  self.assertEqual(m.pred_layer[0].in_features,8)
  with self.assertRaises(ValueError):m.bound_in=True;m.create_model()
 def test_sigmoid_interval_ignores_stale_upper_tensor(self):
  m=model().eval()
  for p in m.parameters():p.data.zero_()
  x=(torch.tensor([[1,2],[2,1]]),[2,2],torch.tensor([1.,2.]),torch.tensor([4.,12.]))
  torch.testing.assert_close(m(x),torch.tensor([52.,56.]))
 def test_lower_bounds_identical_to_paired_rules(self):
  m=model();b=CLIQUEestimator_bound();b.n_db=100;b.substr_library=m.substr_library
  for q in ['aba','aba%','%aba','%aba%','%aba%_','%','%%','%b%','xyz']:
   self.assertEqual(m.get_lower_bound_for_query(q),b.get_lower_bound_for_query(q))
   self.assertEqual(m.get_effective_lower_bound_for_query(q,1),b.get_effective_lower_bound_for_query(q,100))
 def test_cache_upper_endpoints_and_labels_and_edges(self):
  m=model();m.chain_index=[dict(patterns=['aba','ab_','%'],is_anchor=[True,False,True])]
  data=[('aba',4),('ab_',-1),('%',100)]
  m._precompute_chain_pattern_bounds_cache(m.chain_index)
  self.assertTrue(all(ub==100 for ub,lb in m.chain_pattern_bounds_cache.values()))
  batch=m._prepare_chain_batches(data,m.chain_index,CHARS,8)[0]
  self.assertEqual(batch['cards'],[4.,-1.,100.]);self.assertEqual(batch['pair_tight_idx'],[0,1])
  self.assertEqual(batch['pair_loose_idx'],[1,2]);self.assertEqual(batch['upper_bounds'],[100]*3)
  # Exact same chain assembly as paired BOLT with an injected constant interval.
  b=CLIQUEestimator_bound_mono();b.n_db=100;b.chain_pattern_bounds_cache=m.chain_pattern_bounds_cache
  self.assertEqual(batch,b._prepare_chain_batches(data,m.chain_index,CHARS,8)[0])
 def test_reload_and_no_exact_hit_shortcut(self):
  m=model().eval()
  with patch('src.CLIQUE.is_LIKEquery_in_Table',side_effect=AssertionError('UB shortcut called')):
   # Bypass character tokenizer only; exercise the full new prediction entrypoint.
   with patch('src.util.canonicalize_like_query',side_effect=lambda q,**kw:q):
    a=m.estimate(['aba','%aba%'],to_cuda=False)
    f=io.BytesIO();torch.save(m.state_dict(),f);f.seek(0)
    n=model().eval();n.load_state_dict(torch.load(f,weights_only=True))
    self.assertEqual(a,n.estimate(['aba','%aba%'],to_cuda=False))
    self.assertTrue(all(0<=p<=100 for p in a))
 def test_order_gradient_without_hidden_targets(self):
  pred=torch.tensor([30.,10.,40.],requires_grad=True);y=torch.tensor([3.,-1.,40.]);mask=y>=0
  sup=msle_loss(y[mask],pred[mask]);mono=hinge_monotonicity_loss(pred[:2],pred[1:],tolerance=.1)
  loss=sup+5*mono;loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(float(pred.grad[0]),0)
  self.assertLess(float(pred.grad[1]),0)

if __name__=='__main__':unittest.main()

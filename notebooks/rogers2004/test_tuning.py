import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import rogers_model as m
from tune_model import revised_environment, epoch_step

class TuningChecks(unittest.TestCase):
    def test_late_gradient(self):
        from refine_stability import late_bptt
        torch.set_num_threads(1)
        cues,masks,targets=m.epoch_patterns(m.make_environment(),np.random.default_rng(11))
        w=m.SemanticNetwork().double().weight
        cue,target=cues[0].double(),targets[0].double()
        a=m.dynamics(w,cue[None],masks[0][None],56)[21:57,0,:216]
        active=((a.detach()-target).abs()>.01).double()
        loss=(.25*8/36)*(F.binary_cross_entropy(a,target.expand_as(a),reduction='none')*active).sum()
        expected,=torch.autograd.grad(loss,w)
        actual=late_bptt(w.detach(),cue,masks[0],target)
        torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-7)

    def test_tight_margin_gradient(self):
        torch.set_num_threads(1)
        cues,masks,targets=m.epoch_patterns(m.make_environment(),np.random.default_rng(11))
        w=m.SemanticNetwork().double().weight
        cue,target=cues[0].double(),targets[0].double()
        a=m.dynamics(w,cue[None],masks[0][None])[21:29,0,:216]
        active=((a.detach()-target).abs()>.01).double()
        loss=.25*(F.binary_cross_entropy(a,target.clamp(.01,.99).expand_as(a),reduction='none')*active).sum()
        expected,=torch.autograd.grad(loss,w)
        actual,_=m.bptt(w.detach(),cue,masks[0],target,.01)
        torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-7)

    def test_saved_revised_data_survives_reload(self):
        original=m.make_environment(); env=revised_environment()
        np.testing.assert_array_equal(original['targets'][:,:136],env['targets'][:,:136])
        np.testing.assert_array_equal(original['targets'][:,144:],env['targets'][:,144:])
        self.assertTrue(np.any(original['targets'][:,136:144]!=env['targets'][:,136:144]))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            torch.save(dict(state_dict=m.SemanticNetwork().state_dict(),seed=12345,data_seed=2004),root/'model.pt')
            np.savez_compressed(root/'environment.npz',**env)
            _,loaded=m.load(root)
            np.testing.assert_array_equal(loaded['targets'],env['targets'])

if __name__=='__main__': unittest.main()

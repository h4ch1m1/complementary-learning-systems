"""Checks numerical BPTT and task scoring, rather than expected paper trends."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from rogers_model import (SemanticNetwork, make_environment, epoch_patterns,
                          bptt, dynamics, settle, score_tasks, TOTAL)


class ModelChecks(unittest.TestCase):
    def test_bptt_matches_autograd(self):
        torch.set_num_threads(1)
        env = make_environment()
        cues, masks, targets = epoch_patterns(env, np.random.default_rng(11))
        model = SemanticNetwork().double()
        # Include both an untrained and a stronger-weight case.
        for gain in [1., 3.]:
            for i in [0, 1, 2, 3]:
                w = (model.weight.detach() * gain).requires_grad_()
                cue, target = cues[i].double(), targets[i].double()
                h = dynamics(w, cue[None], masks[i][None])
                a = h[21:29, 0, :216]
                active = ((a.detach() - target).abs() > .05).double()
                loss = .25 * (F.binary_cross_entropy(a, target.clamp(.05, .95).expand_as(a), reduction='none') * active).sum()
                expected, = torch.autograd.grad(loss, w)
                actual, _ = bptt(w.detach(), cue, masks[i], target)
                torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
                # Hard clamp is held only during the first three intervals.
                torch.testing.assert_close(h[:12, 0, :216][:, masks[i]], cue[masks[i]].expand(12, -1))

    def test_environment_and_masks(self):
        env = make_environment()
        np.testing.assert_array_equal(env['targets'], make_environment()['targets'])
        self.assertEqual(env['targets'].shape, (48, 216))
        self.assertEqual(len(env['names']), 40)
        self.assertEqual((env['name_ids'] >= 4).sum(), 36)
        self.assertEqual(np.unique(env['targets'][:, 152:], axis=0).shape[0], 48)
        self.assertTrue(np.array_equal(env['targets'][:, 144:150].argmax(1), env['category']))
        model = SemanticNetwork()
        self.assertTrue(torch.all(model.weight[:216, :216] == 0))
        self.assertEqual(int(model.allowed.sum()), 216*64*2 + 64*64)

    def test_streamed_settling_matches_training_dynamics(self):
        env = make_environment()
        cues, masks, _ = epoch_patterns(env, np.random.default_rng(7))
        model = SemanticNetwork()
        with torch.no_grad():
            full = dynamics(model.weight, cues[:3], masks[:3], 28)
            final, _, _ = settle(model.weight, cues[:3], masks[:3], 28)
            torch.testing.assert_close(final, full[-1], atol=0, rtol=0)

    def test_task_scoring_on_perfect_outputs(self):
        env = make_environment()
        out = np.zeros((3, 48, TOTAL), dtype=np.float32)
        out[:, :, :216] = env['targets']
        out[:, :, 216:264] = np.eye(48)
        scores = score_tasks(out, env, out.copy())
        self.assertEqual(scores['naming/all/correct'], 1.)
        for key, value in scores.items():
            if key.startswith(('sorting/', 'matching/')): self.assertEqual(value, 1.)
            if key.startswith(('drawing/', 'delayed_copy/')) and value is not None: self.assertEqual(value, 0.)
        out[:, :, :40] = 0
        scores = score_tasks(out, env, out.copy())
        self.assertEqual(scores['naming/all/omission'], 1.)


if __name__ == '__main__': unittest.main()

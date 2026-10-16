"""Full-vocabulary actors share detached dynamic credit, not actor gradients."""
import unittest

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.dflash.providers import algorithm_providers, algorithm_spec


class TestSixActors(unittest.TestCase):
    def test_offline_capture_keeps_target_final_hidden_state(self):
        name = "target_last_hidden_states"
        self.assertIn(name, algorithm_spec().feature_contracts[0].required_tensors)
        self.assertEqual(
            algorithm_providers().offline[0].capture_layout.last_hidden_feature,
            name,
        )

    def test_formulas_gradients_credit_masks_and_sequence_reduction(self):
        for loss_type in ("dpard", "dptv", "dynamic-ce-tv", "dynamic-nla", "dynamic-lk", "dynamic-kl"):
            with self.subTest(loss_type=loss_type):
                logits = torch.tensor([[[[.2, -.7, 1.], [1., .4, -.3], [.8, -.2, .1]],
                                        [[-.5, .3, .9], [.7, -.8, .2], [.3, .6, -.4]]]],
                                      requires_grad=True)
                teacher = torch.tensor([[[[.9, -.1, .2], [-.6, .8, .3], [.1, .7, -.2]],
                                         [[.8, .1, -.6], [-.1, .5, .7], [.2, -.4, .9]]]],
                                       requires_grad=True)
                ids = torch.tensor([[[0, 2, 1], [2, 0, 1]]])
                mask = torch.tensor([[[0., 1., 1.], [0., 1., 0.]]])
                m = OnlineDFlashModel(nn.Identity(), nn.Identity(), nn.Embedding(3, 3),
                    mask_token_id=2, block_size=3, attention_backend="sdpa",
                    loss_type=loss_type, loss_reduction="sequence_anchor_mean", dpace_alpha=.5)
                p = teacher.detach().softmax(-1)
                logq = logits.log_softmax(-1)
                q = logq.exp()
                tv = .5 * (p - q).abs().sum(-1)
                overlap = torch.minimum(p, q).sum(-1)
                ce = -logq.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                lam = torch.exp(-3 * overlap.detach())
                expected = {"dpard": -2 * (p.sqrt() * q.sqrt()).sum(-1).log(),
                            "dptv": tv, "dynamic-ce-tv": .1 * ce + .9 * tv,
                            "dynamic-nla": -overlap.log(),
                            "dynamic-lk": lam * (p * (p.log() - logq)).sum(-1) + (1-lam)*tv,
                            "dynamic-kl": (p * (p.log() - logq)).sum(-1)}[loss_type]
                smooth = .5 + .5 * overlap.detach()
                credit = torch.zeros_like(mask)
                credit[0, 0, 1] = smooth[0, 0, 1] * (1 + smooth[0, 0, 2])
                credit[0, 0, 2] = smooth[0, 0, 1] * smooth[0, 0, 2]
                credit[0, 1, 1] = smooth[0, 1, 1]
                terms = m._dflash_objective_chunk_terms(logits, ids, mask, ids, teacher,
                                                       m._sequence_anchor_scales(mask))
                want = (expected * credit).sum() / 2
                torch.testing.assert_close(terms.ce_loss_num / terms.loss_den, want)
                actual_grad = torch.autograd.grad(terms.ce_loss_num, logits, retain_graph=True)[0]
                wanted_grad = torch.autograd.grad(want, logits)[0]
                torch.testing.assert_close(actual_grad, wanted_grad)
                self.assertTrue(torch.equal(actual_grad[mask == 0], torch.zeros_like(actual_grad[mask == 0])))
                self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()

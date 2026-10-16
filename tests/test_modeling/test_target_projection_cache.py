"""Teacher projection is shared across objective chunks and backward replay."""
import unittest
from functools import partial

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.dflash import DFlashDraftModel


class TargetProjectionCacheTest(unittest.TestCase):
    def test_cached_chunks_match_uncached_loss_and_gradients(self):
        torch.manual_seed(42)
        for kind in ("dpard", "dynamic-kl", "dynamic-nla", "dptv", "dynamic-ce-tv", "dynamic-lk"):
            model = OnlineDFlashModel(
                nn.Identity(), nn.Linear(7, 11, bias=False).requires_grad_(False),
                nn.Embedding(11, 7), mask_token_id=10, block_size=4,
                attention_backend="sdpa", loss_type=kind,
                loss_reduction="sequence_anchor_mean", dpace_alpha=0.3,
            )
            hidden = torch.randn(2, 3, 4, 7, requires_grad=True)
            teacher = torch.randn(2, 9, 7, requires_grad=True)
            # Repeated and boundary positions exercise the chunk-local gather.
            indices = torch.tensor([[[0, 0, 1, 2], [2, 3, 4, 5], [5, 6, 7, 8]]] * 2)
            aligned = teacher[:, None].expand(-1, 3, -1, -1).gather(
                2, indices[..., None].expand(-1, -1, -1, 7)
            )
            ids = torch.randint(11, (2, 3, 4))
            for empty in (False, True):
                mask = torch.ones_like(ids, dtype=torch.float)
                mask[..., 0] = 0
                mask[1] = 0
                if empty:
                    mask.zero_()
                scales = model._sequence_anchor_scales(mask)
                reference = model._dflash_objective_chunk_terms(
                    hidden, ids, mask, ids, aligned, scales,
                )
                reference_grad = torch.autograd.grad(reference.ce_loss_num, hidden)[0]
                with torch.no_grad():
                    cached_logits = model.lm_head(teacher)
                for chunk in (0, 1, 2):
                    with self.subTest(kind=kind, chunk=chunk, empty=empty):
                        terms = checkpointed_chunk_reduce(
                            partial(model._dflash_objective_chunk_terms, target_logits=cached_logits),
                            hidden, ids, mask, ids, None, scales, indices,
                            chunk_size=chunk, dim=1,
                        )
                        for actual, expected in zip(terms, reference):
                            torch.testing.assert_close(actual, expected)
                        grad = torch.autograd.grad(terms[0], hidden)[0]
                        torch.testing.assert_close(grad, reference_grad)
                        self.assertIsNone(teacher.grad)

    def test_forward_backward_projects_teacher_once(self):
        for kind in ("dpard", "dynamic-kl", "dynamic-nla"):
            torch.manual_seed(42)
            config = Qwen3Config(
                architectures=["DFlashDraftModel"], hidden_size=32,
                intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                num_hidden_layers=3, num_target_layers=36, head_dim=8,
                max_position_embeddings=128, vocab_size=64,
                layer_types=["full_attention"] * 3,
                dflash_config={"block_size": 16, "target_layer_ids": [1, 17, 33]},
            )
            config._attn_implementation = "sdpa"
            model = OnlineDFlashModel(
                DFlashDraftModel(config),
                nn.Linear(32, 64, bias=False).requires_grad_(False),
                nn.Embedding(64, 32).requires_grad_(False),
                mask_token_id=63, block_size=16, num_anchors=3,
                attention_backend="sdpa", loss_type=kind,
                loss_reduction="sequence_anchor_mean", dpace_alpha=0.3,
            )
            ids = torch.randint(0, 63, (2, 48))
            features = torch.randn(2, 48, 96)
            teacher = torch.randn(2, 48, 32, requires_grad=True)
            mask = torch.ones_like(ids, dtype=torch.float)
            mask[1] = 0  # An empty sequence must not change normalization.
            reference = None
            for chunk in (0, 1, 2):
                for detailed in (False, True):
                    with self.subTest(kind=kind, chunk=chunk, detailed=detailed):
                        projections = []
                        def record_projection(module, args):
                            if not torch.is_grad_enabled():
                                projections.append(tuple(args[0].shape))
                        hook = model.lm_head.register_forward_pre_hook(record_projection)
                        model.objective_chunk_blocks = chunk
                        model.zero_grad(set_to_none=True)
                        torch.manual_seed(7)
                        try:
                            loss, _, _ = model(
                                ids, features, mask, target_last_hidden_states=teacher,
                                collect_detailed_metrics=detailed,
                            )
                            loss.backward()
                        finally:
                            hook.remove()
                        self.assertTrue(torch.isfinite(loss))
                        self.assertIsNone(teacher.grad)
                        self.assertTrue(all(p.grad is None for p in model.lm_head.parameters()))
                        grads = torch.cat([
                            p.grad.flatten() for p in model.draft_model.parameters()
                            if p.grad is not None
                        ])
                        self.assertTrue(torch.isfinite(grads).all())
                        self.assertGreater(grads.abs().sum().item(), 0)
                        if reference is None:
                            reference = (loss.detach(), grads.detach().clone())
                        torch.testing.assert_close(loss.detach(), reference[0])
                        torch.testing.assert_close(grads, reference[1], atol=2e-6, rtol=2e-5)
                        if not detailed:
                            self.assertEqual(projections, [(2, 48, 32)])


if __name__ == "__main__":
    unittest.main()

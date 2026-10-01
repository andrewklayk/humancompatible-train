"""
Gate mathematics, density constraints and token sharding for E3.

E3 itself registers no predictions — it is descriptive by design — so the falsifiable
content lives here instead. The checks that matter are the ones a plausible-looking
mistake would pass: the closed-form ``P(z != 0)`` is verified against a Monte-Carlo
estimate rather than against itself, and the parameter-weighted density is verified
against the mean open probability in the one case (equally sized gates) where the two must
coincide.

Runs on CPU with no ``transformers`` install, using the stand-in model in
``paper/problems/tiny_lm.py``.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from paper.problems.sparsity import sparse_lm
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paper.problems import tokens as tokens_mod
from paper.problems.sparsity.sparsity_gates import BETA, GAMMA, ZETA, HardConcreteGate
from paper.problems.sparsity.sparse_lm import attach_gates
from paper.problems.tiny_lm import tiny_causal_lm


class TestHardConcreteGate(unittest.TestCase):
    def test_open_prob_matches_monte_carlo(self):
        """The closed form must match a sampled estimate of P(z != 0).

        This is the check with real discriminating power: the formula
        ``sigmoid(log_alpha - beta*log(-gamma/zeta))`` is easy to transcribe with the
        wrong sign or a missing beta, and every such variant is still a smooth
        probability-shaped function of log_alpha.
        """
        torch.manual_seed(0)
        values = torch.linspace(-3.0, 3.0, 13)
        repeats = 40_000
        gate = HardConcreteGate(values.numel() * repeats)
        with torch.no_grad():
            gate.log_alpha.copy_(values.repeat_interleave(repeats))

        z = gate.sample(torch.Generator().manual_seed(1))
        empirical = (z > 0).double().view(values.numel(), repeats).mean(dim=1)
        closed = gate.open_prob().view(values.numel(), repeats)[:, 0].double()

        # 3 sigma of a Bernoulli mean at n=40000 is at most 3*0.5/200 = 0.0075.
        torch.testing.assert_close(empirical, closed, atol=0.008, rtol=0.0)

    def test_open_shift_constant(self):
        expected = -BETA * math.log(-GAMMA / ZETA)
        self.assertAlmostEqual(sparse_lm._OPEN_SHIFT, expected, places=12)
        # (2/3)*log(11), spelled out so a transcription slip in either direction shows up.
        self.assertAlmostEqual(sparse_lm._OPEN_SHIFT, 1.5985968, places=6)

    def test_init_open_is_honoured(self):
        for target in (0.5, 0.9, 0.99):
            gate = HardConcreteGate(4096, init_open=target, init_std=0.0)
            self.assertAlmostEqual(float(gate.open_prob().mean()), target, places=5)

    def test_median_is_the_sample_median(self):
        """`median()` must be the 50th percentile of `sample()`, not merely close to it."""
        torch.manual_seed(0)
        gate = HardConcreteGate(7, init_open=0.7, init_std=1.0)
        draws = torch.stack(
            [gate.sample(torch.Generator().manual_seed(s)) for s in range(4001)]
        )
        torch.testing.assert_close(
            draws.median(dim=0).values, gate.median(), atol=2e-2, rtol=0.0
        )

    def test_median_formula_includes_the_temperature(self):
        """arXiv:2208.04425 uses medians, so the /beta must be there.

        Louizos et al. Eq. (13) writes the estimator without it; the two differ whenever
        log_alpha is not 0, and this pins which one we implement.
        """
        gate = HardConcreteGate(3, init_std=0.0)
        with torch.no_grad():
            gate.log_alpha.copy_(torch.tensor([-1.0, 0.0, 2.0]))
        expected = (
            torch.sigmoid(gate.log_alpha / BETA) * (ZETA - GAMMA) + GAMMA
        ).clamp(0, 1)
        torch.testing.assert_close(gate.median(), expected)
        without_beta = (torch.sigmoid(gate.log_alpha) * (ZETA - GAMMA) + GAMMA).clamp(0, 1)
        self.assertFalse(torch.allclose(gate.median(), without_beta))

    def test_samples_are_in_range_and_hit_both_ends(self):
        """The point of the hard concrete: exact zeros and exact ones both occur."""
        torch.manual_seed(0)
        gate = HardConcreteGate(20_000, init_open=0.5, init_std=1.0)
        z = gate.sample(torch.Generator().manual_seed(3))
        self.assertGreaterEqual(float(z.min()), 0.0)
        self.assertLessEqual(float(z.max()), 1.0)
        self.assertGreater(int((z == 0).sum()), 0)
        self.assertGreater(int((z == 1).sum()), 0)

    def test_sample_is_differentiable_in_log_alpha(self):
        gate = HardConcreteGate(64, init_open=0.5)
        z = gate.sample(torch.Generator().manual_seed(0))
        z.sum().backward()
        self.assertIsNotNone(gate.log_alpha.grad)
        self.assertGreater(float(gate.log_alpha.grad.abs().sum()), 0.0)

    def test_initialisation_is_seeded_and_device_independent(self):
        a = HardConcreteGate(16, generator=torch.Generator().manual_seed(5))
        b = HardConcreteGate(16, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(a.log_alpha, b.log_alpha, atol=0.0, rtol=0.0)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            HardConcreteGate(0)
        with self.assertRaises(ValueError):
            HardConcreteGate(4, init_open=1.0)


class TestDensityConstraints(unittest.TestCase):
    def setUp(self):
        self.model = tiny_causal_lm(seed=0, num_hidden_layers=2)
        self.gates = attach_gates(self.model, seed=0)

    def test_granularity_shapes(self):
        n_layers = self.model.config.num_hidden_layers
        self.assertEqual(self.gates.m("model"), 1)
        self.assertEqual(self.gates.m("layer"), n_layers)
        self.assertEqual(self.gates.m("layer_split"), 2 * n_layers)
        for granularity in sparse_lm.GRANULARITIES:
            c = self.gates.constraints(0.5, granularity)
            self.assertEqual(c.shape, (self.gates.m(granularity),))
            self.assertEqual(
                len(self.gates.constraint_names(granularity)), self.gates.m(granularity)
            )

    def test_density_is_a_probability(self):
        for granularity in sparse_lm.GRANULARITIES:
            density = self.gates.densities(granularity).detach()
            self.assertGreaterEqual(float(density.min()), 0.0)
            self.assertLessEqual(float(density.max()), 1.0)

    def test_all_open_gates_give_density_one(self):
        with torch.no_grad():
            for group in self.gates.groups:
                group.gate.log_alpha.fill_(40.0)
        density = self.gates.densities("model")
        torch.testing.assert_close(density, torch.ones(1), atol=1e-6, rtol=0.0)

    def test_eps_one_is_vacuous(self):
        """Their section 3.1: eps >= 1 cannot bind, since density never exceeds 1."""
        with torch.no_grad():
            for group in self.gates.groups:
                group.gate.log_alpha.fill_(40.0)
        for granularity in sparse_lm.GRANULARITIES:
            self.assertLessEqual(
                float(self.gates.constraints(1.0, granularity).detach().max()), 1e-6
            )

    def test_parameter_weighting_collapses_when_gates_are_equal_sized(self):
        """With one gate group per cell the weights cancel, so density is the mean
        open probability. Any weighting bug that survives this is at least consistent."""
        cell_density = self.gates.densities("layer_split")
        names = self.gates.constraint_names("layer_split")
        for group in self.gates.groups:
            index = names.index(group.name)
            torch.testing.assert_close(
                cell_density[index], group.gate.open_prob().mean()
            )

    def test_parameter_weighting_is_not_a_plain_mean_across_groups(self):
        """An MLP gate and a head gate cover different parameter counts, so a fused
        layer cell must not be the unweighted mean of the two group means."""
        with torch.no_grad():
            for group in self.gates.groups:
                group.gate.log_alpha.fill_(2.0 if group.kind == "mlp" else -2.0)
        layer0 = self.gates.densities("layer")[0]
        groups0 = [g for g in self.gates.groups if g.layer == 0]
        unweighted = sum(float(g.gate.open_prob().detach().mean()) for g in groups0) / len(groups0)
        weighted = sum(
            float(g.gate.open_prob().detach().sum()) * g.params_per_gate
            for g in groups0
        ) / sum(g.params_total for g in groups0)
        self.assertAlmostEqual(float(layer0), weighted, places=6)
        self.assertNotAlmostEqual(float(layer0), unweighted, places=3)

    def test_per_gate_parameter_counts(self):
        config = self.model.config
        for group in self.gates.groups:
            if group.kind == "mlp":
                self.assertEqual(group.params_per_gate, 3 * config.hidden_size)
                self.assertEqual(group.n_gates, config.intermediate_size)
                self.assertEqual(group.repeat, 1)
            else:
                self.assertEqual(
                    group.params_per_gate, 2 * config.head_dim * config.hidden_size
                )
                self.assertEqual(group.n_gates, config.num_attention_heads)
                self.assertEqual(group.repeat, config.head_dim)

    def test_constraints_are_differentiable_in_the_gates(self):
        self.gates.constraints(0.5, "model").sum().backward()
        for group in self.gates.groups:
            self.assertIsNotNone(group.gate.log_alpha.grad)
            self.assertGreater(float(group.gate.log_alpha.grad.abs().sum()), 0.0)

    def test_constraints_do_not_depend_on_the_gate_sample(self):
        """The closed form is why the constraint carries no minibatch or sampling noise —
        and also why a data-parallel reduction over it has nothing to pool."""
        before = self.gates.constraints(0.5, "layer").detach().clone()
        self.gates.resample(torch.Generator().manual_seed(11))
        after = self.gates.constraints(0.5, "layer").detach()
        torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)

    def test_per_cell_eps_vector(self):
        eps = [0.3, 0.7]
        c = self.gates.constraints(eps, "layer")
        density = self.gates.densities("layer")
        torch.testing.assert_close(c, density - torch.tensor(eps))
        with self.assertRaises(ValueError):
            self.gates.constraints([0.3], "layer")

    def test_median_report_counts_purged_parameters(self):
        with torch.no_grad():
            for group in self.gates.groups:
                group.gate.log_alpha.fill_(-40.0)  # every median gate closes
        for row in self.gates.median_report():
            self.assertEqual(row["median_density"], 0.0)
            self.assertEqual(row["params_active"], 0)
        with torch.no_grad():
            for group in self.gates.groups:
                group.gate.log_alpha.fill_(40.0)
        for row in self.gates.median_report():
            self.assertEqual(row["median_density"], 1.0)
            self.assertEqual(row["params_active"], row["params_total"])


class TestAttachGates(unittest.TestCase):
    def test_gates_are_model_parameters(self):
        model = tiny_causal_lm(seed=0)
        before = {id(p) for p in model.parameters()}
        gates = attach_gates(model, seed=0)
        after = {id(p) for p in model.parameters()}
        added = after - before
        self.assertEqual(len(added), len(gates.groups))
        # This is what makes DistributedDataParallel synchronise them like any other
        # parameter, and hence what makes every rank agree on the constraint value.
        self.assertTrue(all(id(p) in after for p in gates.gate_parameters()))

    def test_closed_gates_change_the_output(self):
        torch.manual_seed(0)
        model = tiny_causal_lm(seed=0)
        ids = torch.randint(0, model.config.vocab_size, (2, 16))
        gates = attach_gates(model, seed=0)
        gates.use_open()
        open_logits = model(input_ids=ids).logits.detach().clone()
        with torch.no_grad():
            for group in gates.groups:
                group.gate.log_alpha.fill_(-40.0)
        gates.use_median()
        closed_logits = model(input_ids=ids).logits.detach()
        self.assertFalse(torch.allclose(open_logits, closed_logits))

    def test_open_gates_are_inert(self):
        """All-ones gates must reproduce the ungated model exactly, so the hook itself
        introduces no drift and can serve as the timing reference."""
        torch.manual_seed(0)
        plain = tiny_causal_lm(seed=0)
        ids = torch.randint(0, plain.config.vocab_size, (2, 16))
        reference = plain(input_ids=ids).logits.detach().clone()
        gates = attach_gates(plain, seed=0)
        gates.use_open()
        torch.testing.assert_close(plain(input_ids=ids).logits.detach(), reference)

    def test_loss_gradient_reaches_the_gates(self):
        model = tiny_causal_lm(seed=0)
        gates = attach_gates(model, seed=0)
        gates.resample(torch.Generator().manual_seed(0))
        ids = torch.randint(0, model.config.vocab_size, (2, 16))
        model(input_ids=ids, labels=ids).loss.backward()
        for group in gates.groups:
            self.assertIsNotNone(group.gate.log_alpha.grad)
            self.assertGreater(float(group.gate.log_alpha.grad.abs().sum()), 0.0)

    def test_forward_without_a_sample_raises_clearly(self):
        model = tiny_causal_lm(seed=0)
        gates = attach_gates(model, seed=0)
        for group in gates.groups:
            group.z = None
        ids = torch.randint(0, model.config.vocab_size, (1, 8))
        with self.assertRaisesRegex(RuntimeError, "no current sample"):
            model(input_ids=ids)

    def test_head_gate_expands_over_head_dim(self):
        model = tiny_causal_lm(seed=0)
        gates = attach_gates(model, seed=0)
        gates.use_open()
        for group in gates.groups:
            if group.kind != "attn":
                continue
            expanded = group.expanded_z()
            o_proj = sparse_lm.decoder_blocks(model)[group.layer].self_attn.o_proj
            self.assertEqual(expanded.numel(), o_proj.in_features)

    def test_gate_only_flags(self):
        mlp_only = attach_gates(tiny_causal_lm(seed=0), gate_heads=False, seed=0)
        self.assertTrue(all(g.kind == "mlp" for g in mlp_only.groups))
        heads_only = attach_gates(tiny_causal_lm(seed=0), gate_mlp=False, seed=0)
        self.assertTrue(all(g.kind == "attn" for g in heads_only.groups))
        with self.assertRaises(ValueError):
            attach_gates(tiny_causal_lm(seed=0), gate_mlp=False, gate_heads=False)

    def test_double_attach_is_refused(self):
        model = tiny_causal_lm(seed=0)
        attach_gates(model, seed=0)
        with self.assertRaises(ValueError):
            attach_gates(model, seed=0)

    def test_remove_restores_the_ungated_forward(self):
        torch.manual_seed(0)
        model = tiny_causal_lm(seed=0)
        ids = torch.randint(0, model.config.vocab_size, (2, 16))
        reference = model(input_ids=ids).logits.detach().clone()
        gates = attach_gates(model, seed=0)
        with torch.no_grad():
            for group in gates.groups:
                group.gate.log_alpha.fill_(-40.0)
        gates.use_median()
        self.assertFalse(torch.allclose(model(input_ids=ids).logits.detach(), reference))
        gates.remove()
        torch.testing.assert_close(model(input_ids=ids).logits.detach(), reference)

    def test_describe_reports_the_ungated_share(self):
        model = tiny_causal_lm(seed=0)
        gates = attach_gates(model, seed=0)
        facts = sparse_lm.describe(model, gates)
        self.assertEqual(facts["gate_params"], sum(len(g.gate) for g in gates.groups))
        # Embeddings and norms are not gated, so the reachable share is strictly below 1.
        # That is the reason the density denominator runs over gated groups only.
        self.assertGreater(facts["gated_fraction"], 0.0)
        self.assertLess(facts["gated_fraction"], 1.0)

    def test_rejects_a_model_without_blocks(self):
        with self.assertRaises(TypeError):
            sparse_lm.decoder_blocks(torch.nn.Linear(2, 2))


class TestTokenShard(unittest.TestCase):
    def test_round_trip_and_block_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.bin"
            ids = np.arange(100, dtype=np.int64) % 50
            written = tokens_mod.write_shard(path, ids, vocab_size=50)
            self.assertEqual(written, 100)
            shard = tokens_mod.TokenShard(path, seq_len=8)
            self.assertEqual(shard.n_tokens, 100)
            self.assertEqual(len(shard), 12)  # 100 // 8, remainder dropped
            np.testing.assert_array_equal(shard.block(0), ids[:8])
            batch = shard.batch([0, 1])
            self.assertEqual(batch.shape, (2, 8))
            self.assertEqual(batch.dtype, torch.int64)

    def test_token_width_follows_the_vocabulary(self):
        """SmolLM's 49152 fits uint16; Qwen2.5's 151936 does not and must widen.

        A uint16 shard reread for a 151936-token vocabulary would wrap silently and look
        like an unexplained loss plateau, so the width is recorded rather than assumed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            narrow = Path(tmp) / "narrow.bin"
            tokens_mod.write_shard(narrow, [0, 49_151], vocab_size=49_152)
            self.assertEqual(tokens_mod.TokenShard(narrow, 2).dtype, np.dtype(np.uint16))

            wide = Path(tmp) / "wide.bin"
            tokens_mod.write_shard(wide, [0, 151_935], vocab_size=151_936)
            shard = tokens_mod.TokenShard(wide, 2)
            self.assertEqual(shard.dtype, np.dtype(np.uint32))
            np.testing.assert_array_equal(shard.block(0), [0, 151_935])

            # A small-id shard still widens if the declared vocabulary demands it.
            declared = Path(tmp) / "declared.bin"
            tokens_mod.write_shard(declared, [0, 1], vocab_size=151_936)
            self.assertEqual(
                tokens_mod.TokenShard(declared, 2).dtype, np.dtype(np.uint32)
            )

    def test_rejects_bad_writes_and_a_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.bin"
            with self.assertRaises(ValueError):
                tokens_mod.write_shard(path, [])
            with self.assertRaises(ValueError):
                tokens_mod.write_shard(path, [0, -1])
            tokens_mod.write_shard(path, [0, 1, 2, 3], vocab_size=4)
            Path(str(path) + ".meta.json").unlink()
            with self.assertRaisesRegex(ValueError, "token width is unknown"):
                tokens_mod.TokenShard(path, 2)
            # ...unless the caller says what it is.
            self.assertEqual(len(tokens_mod.TokenShard(path, 2, dtype=np.uint16)), 2)

    def test_missing_shard_points_at_prepare_data(self):
        with self.assertRaisesRegex(FileNotFoundError, "prepare_data"):
            tokens_mod.TokenShard("/nonexistent/tokens.bin", 8)

    def test_synthetic_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.bin"
            tokens_mod.write_synthetic_shard(path, 512, vocab_size=97, seed=1)
            shard = tokens_mod.TokenShard(path, seq_len=16)
            self.assertEqual(shard.n_tokens, 512)
            self.assertLess(int(shard.block(0).max()), 97)


class TestBlockSampler(unittest.TestCase):
    def test_ranks_are_disjoint_and_equal_length(self):
        """Equal length is the load-bearing property: a rank that runs out of batches
        early stops calling collectives and the job hangs rather than failing."""
        world, n_blocks, batch = 3, 100, 4
        samplers = [
            tokens_mod.BlockSampler(n_blocks, batch, rank=r, world=world, seed=0)
            for r in range(world)
        ]
        lengths = {len(s) for s in samplers}
        self.assertEqual(lengths, {100 // 3 // 4})
        seen = []
        for sampler in samplers:
            for indices in sampler.epoch(0):
                seen.extend(int(i) for i in indices)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertTrue(all(0 <= i < n_blocks for i in seen))

    def test_epoch_order_is_seeded_and_epoch_dependent(self):
        first = tokens_mod.BlockSampler(64, 2, seed=7)
        second = tokens_mod.BlockSampler(64, 2, seed=7)
        a = [i.tolist() for i in first.epoch(0)]
        b = [i.tolist() for i in second.epoch(0)]
        self.assertEqual(a, b)
        c = [i.tolist() for i in first.epoch(1)]
        self.assertNotEqual(a, c)

    def test_stream_yields_exactly_the_requested_steps(self):
        sampler = tokens_mod.BlockSampler(32, 4, seed=0)  # 8 batches per epoch
        batches = list(sampler.stream(20))
        self.assertEqual(len(batches), 20)
        self.assertTrue(all(len(b) == 4 for b in batches))

    def test_rejects_impossible_configurations(self):
        with self.assertRaises(ValueError):
            tokens_mod.BlockSampler(4, 8, seed=0)  # no complete batch
        with self.assertRaises(ValueError):
            tokens_mod.BlockSampler(64, 2, rank=3, world=3)
        with self.assertRaises(NotImplementedError):
            tokens_mod.BlockSampler(64, 2, drop_last=False)


if __name__ == "__main__":
    unittest.main()

"""Equivalence tests for the Apple-Silicon (MPS) code paths and the sync-free Lamb."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torch_brain.nn import Embedding, InfiniteVocabEmbedding

mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an Apple-Silicon GPU"
)


def test_infinite_vocab_mps_path_matches_reference():
    torch.manual_seed(0)
    emb = InfiniteVocabEmbedding(16)
    emb.initialize_vocab([f"unit_{i}" for i in range(50)])
    idx = torch.randint(1, 51, (4, 64))
    idx[:, 40:] = emb.padding_idx  # heavy padding, like real batches

    ref_w = emb.weight.detach().clone().requires_grad_(True)
    new_w = emb.weight.detach().clone().requires_grad_(True)
    grad = torch.randn(4, 64, 16)

    ref = F.embedding(idx, ref_w, emb.padding_idx)
    emb.weight = torch.nn.Parameter(new_w)
    new = emb._forward_mps(idx)  # the MPS code path also runs on the CPU
    torch.testing.assert_close(new, ref)

    ref.backward(grad)
    new.backward(grad)
    torch.testing.assert_close(emb.weight.grad, ref_w.grad)


@mps
def test_infinite_vocab_on_mps_matches_cpu():
    torch.manual_seed(0)
    emb = InfiniteVocabEmbedding(16)
    emb.initialize_vocab([f"unit_{i}" for i in range(50)])
    idx = torch.randint(0, 51, (4, 64))
    cpu = emb(idx)
    gpu = copy.deepcopy(emb).to("mps")(idx.to("mps")).cpu()
    torch.testing.assert_close(gpu, cpu)


@mps
def test_small_embedding_one_hot_on_mps_matches_cpu():
    torch.manual_seed(0)
    emb = Embedding(4, 16)
    idx = torch.randint(0, 4, (8, 128))
    grad = torch.randn(8, 128, 16)

    out_cpu = emb(idx)
    out_cpu.backward(grad)

    emb_mps = copy.deepcopy(emb).to("mps")
    emb_mps.weight.grad = None
    out_mps = emb_mps(idx.to("mps"))
    out_mps.backward(grad.to("mps"))

    torch.testing.assert_close(out_mps.cpu(), out_cpu)
    torch.testing.assert_close(emb_mps.weight.grad.cpu(), emb.weight.grad, atol=1e-4, rtol=1e-4)


def test_sync_free_lamb_matches_torch_optimizer():
    torch_optimizer = pytest.importorskip("torch_optimizer")
    from torch_brain.optim import Lamb

    torch.manual_seed(0)
    m1 = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 2))
    with torch.no_grad():
        m1[2].bias.zero_()  # a zero-norm parameter exercises the trust-ratio branch
    m2 = copy.deepcopy(m1)
    o1 = torch_optimizer.Lamb(m1.parameters(), lr=2e-3, weight_decay=1e-4)
    o2 = Lamb(m2.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(20):
        x = torch.randn(32, 8)
        for model, opt in ((m1, o1), (m2, o2)):
            opt.zero_grad()
            model(x).pow(2).mean().backward()
            opt.step()
    for a, b in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_poyo_plus_tokenizer_emits_float32_timestamps():
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    from torch_brain.models import POYOPlus, POYOPlusTokenizer
    from torch_brain.registry import MODALITIY_REGISTRY

    model = POYOPlus(dim=32, dim_head=16, num_latents=4, depth=1, readout_specs=MODALITIY_REGISTRY)
    model.unit_emb.initialize_vocab(["u0", "u1"])
    model.session_emb.initialize_vocab(["s0"])
    tok = POYOPlusTokenizer(
        model.unit_emb.tokenizer,
        model.session_emb.tokenizer,
        decoder_registry=MODALITIY_REGISTRY,
        latent_step=0.25,
        num_latents_per_step=4,
    )
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.5, 0.9]), unit_index=np.array([0, 1, 0]), domain="auto"
        ),
        units=ArrayDict(id=np.array(["u0", "u1"])),
        cursor=IrregularTimeSeries(
            timestamps=np.linspace(0, 1, 10), vel=np.random.randn(10, 2), domain="auto"
        ),
        session="s0",
        config={"multitask_readout": [{"readout_id": "cursor_velocity_2d"}]},
        domain=Interval(0.0, 1.0),
    )
    batch = tok(data)
    assert batch["input_timestamps"].obj.dtype == np.float32
    assert batch["latent_timestamps"].dtype == np.float32
    assert batch["output_timestamps"].obj.dtype == torch.float32

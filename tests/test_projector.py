"""GenomicVirtualTokenProjector -- shapes and the missing-modality path. CPU only."""

from __future__ import annotations

import pytest
import torch

from utils.token_projector import GenomicVirtualTokenProjector

LLAMA_HIDDEN = 4096
NUM_VIRTUAL = 8


@pytest.fixture(params=[("dna", 4096), ("esm", 2560), ("pst", 1280)], ids=lambda p: p[0])
def projector(request):
    _, dim = request.param
    torch.manual_seed(0)
    return GenomicVirtualTokenProjector(
        embedding_dim=dim,
        llama_hidden_size=LLAMA_HIDDEN,
        num_virtual_tokens=NUM_VIRTUAL,
    ), dim


def test_projects_to_virtual_tokens(projector):
    model, dim = projector
    out = model(torch.randn(1, dim))
    assert out.shape == (1, NUM_VIRTUAL, LLAMA_HIDDEN)
    assert torch.isfinite(out).all()


def test_null_embedding_used_when_modality_missing(projector):
    """Called with no embedding, the projector falls back to its learnable null."""
    model, _ = projector
    out = model()
    assert out.shape[-2:] == (NUM_VIRTUAL, LLAMA_HIDDEN)
    assert torch.isfinite(out).all()


def test_batch_size_argument_expands_the_null_embedding(projector):
    model, _ = projector
    out = model(batch_size=4)
    assert out.shape == (4, NUM_VIRTUAL, LLAMA_HIDDEN)


def test_null_embedding_is_trainable(projector):
    """It has to receive gradients, or missing modalities never learn a representation."""
    model, _ = projector
    model().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient from the null path"
    assert any(g.abs().sum() > 0 for g in grads)


def test_output_depends_on_input(projector):
    model, dim = projector
    model.eval()
    with torch.no_grad():
        a = model(torch.randn(1, dim))
        b = model(torch.randn(1, dim))
    assert not torch.allclose(a, b), "projector output is independent of its input"

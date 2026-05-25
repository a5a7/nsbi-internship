"""Tests for build_classifier_ssl / _freeze_backbone in nsbi/train.py.

Run from the EveNet-Lite repo root:
    python -m pytest tests/test_ssl_classifier.py -v
"""

import logging

import pytest
import torch

from evenet_lite import EvenetLiteClassifier
from nsbi.train import _freeze_backbone, SEQ_DIM, GLOBAL_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clf(pretrained: bool = False) -> EvenetLiteClassifier:
    """Build a minimal CPU classifier — no HuggingFace download needed."""
    return EvenetLiteClassifier(
        class_labels=["bkg", "sig"],
        device="cpu",
        lr=[1e-3],
        weight_decay=1e-2,
        module_lists=[["Classification"]],
        sequential_input_dim=SEQ_DIM,
        global_input_dim=GLOBAL_DIM,
        pretrained=pretrained,
        log_level=logging.WARNING,  # suppress logs during tests
    )


def _get_members(clf: EvenetLiteClassifier):
    """Return list of _EveNetLiteSingle members regardless of ensemble mode."""
    model = clf.model
    if hasattr(model, "models"):
        return list(model.models)
    return [model]


def _count_trainable(clf: EvenetLiteClassifier) -> int:
    return sum(p.numel() for p in clf.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Test 1: backbone modules are frozen after _freeze_backbone
# ---------------------------------------------------------------------------

def test_backbone_modules_are_frozen():
    """PET, GlobalEmbedding, ObjectEncoder must all have requires_grad=False."""
    clf = _make_clf()
    _freeze_backbone(clf)

    for member in _get_members(clf):
        backbone = getattr(member, "backbone", None)
        assert backbone is not None, "Member has no backbone attribute"

        for module_name in ("PET", "GlobalEmbedding", "ObjectEncoder"):
            module = getattr(backbone, module_name, None)
            assert module is not None, f"Backbone missing module: {module_name}"
            for param_name, p in module.named_parameters():
                assert not p.requires_grad, (
                    f"{module_name}.{param_name} should be frozen "
                    f"but requires_grad=True"
                )


# ---------------------------------------------------------------------------
# Test 2: Classification head remains trainable
# ---------------------------------------------------------------------------

def test_classification_head_remains_trainable():
    """Classification head parameters must stay requires_grad=True."""
    clf = _make_clf()
    _freeze_backbone(clf)

    trainable = 0
    for member in _get_members(clf):
        head = getattr(member, "Classification", None)
        assert head is not None, "Member has no Classification attribute"
        for param_name, p in head.named_parameters():
            assert p.requires_grad, (
                f"Classification.{param_name} should be trainable "
                f"but requires_grad=False"
            )
            trainable += p.numel()

    assert trainable > 0, "Classification head has zero trainable parameters"


# ---------------------------------------------------------------------------
# Test 3: SSL variant has fewer trainable params than full fine-tuning
# ---------------------------------------------------------------------------

def test_ssl_has_fewer_trainable_params_than_full():
    """Frozen backbone means significantly fewer trainable parameters."""
    clf_full = _make_clf()                  # all layers trainable
    clf_ssl  = _make_clf()
    _freeze_backbone(clf_ssl)               # backbone frozen

    n_full = _count_trainable(clf_full)
    n_ssl  = _count_trainable(clf_ssl)

    assert n_ssl > 0, "SSL classifier has no trainable parameters at all"
    assert n_ssl < n_full, (
        f"SSL should have fewer trainable params than full fine-tuning "
        f"(got ssl={n_ssl}, full={n_full})"
    )
    # Classification head should be a small fraction of the total model
    assert n_ssl < n_full * 0.5, (
        f"Expected SSL trainable params to be <50% of full "
        f"(got {n_ssl / n_full:.1%})"
    )


# ---------------------------------------------------------------------------
# Test 4: backbone weights literally do not change after a backward pass
# ---------------------------------------------------------------------------

def test_backbone_weights_unchanged_after_backward():
    """Run a real forward+backward pass and confirm backbone weights are frozen."""
    clf = _make_clf()
    _freeze_backbone(clf)

    model = clf.model
    model.train()

    # snapshot backbone weights before the pass
    def _snapshot(member):
        backbone = member.backbone
        return {
            name: p.detach().clone()
            for name, p in backbone.named_parameters()
        }

    members = _get_members(clf)
    before = [_snapshot(m) for m in members]

    # build a tiny random batch matching NSBI dims:
    #   x       : [B, 4 leptons, SEQ_DIM features]
    #   x_mask  : [B, 4]  all True
    #   globals : [B, GLOBAL_DIM]
    B = 4
    x       = torch.randn(B, 4, SEQ_DIM)
    x_mask  = torch.ones(B, 4, dtype=torch.bool)
    globals_ = torch.randn(B, GLOBAL_DIM)

    logits = model(x, x_mask, globals_)          # forward pass
    loss   = logits.float().sum()                 # trivial scalar loss
    loss.backward()                               # backward pass

    # manually step the Classification parameters (simulate optimizer)
    with torch.no_grad():
        for member in members:
            for p in member.Classification.parameters():
                if p.grad is not None:
                    p -= 1e-3 * p.grad             # tiny SGD step

    # now verify backbone weights haven't moved at all
    for member, snap in zip(members, before):
        backbone = member.backbone
        for name, p in backbone.named_parameters():
            assert torch.equal(p.detach(), snap[name]), (
                f"Backbone weight {name} changed after backward pass — "
                f"freeze is not holding!"
            )

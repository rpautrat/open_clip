"""Cone-effect eval (--cone-val): spread-out and uniformity metrics."""
import math

import pytest
import torch
from torchvision import transforms

from open_clip.task import CLIPTask
from open_clip_train.cone_eval import (
    build_cone_datasets,
    cone_eval,
    spread_out_metric,
    uniformity_metric,
)
from open_clip_train.train import zero_shot_eval_all
from test_retrieval_eval import _args, _tokenizer, _write_shards
from util_test import create_tiny_model


def _bruteforce_spread_out(features):
    features = torch.nn.functional.normalize(features.double(), dim=-1)
    sims = features @ features.T
    off_diagonal = ~torch.eye(len(features), dtype=torch.bool)
    return sims[off_diagonal].mean().item()


def _reference_uniformity(features, t):
    """Reference implementation from Wang & Isola (2020), over unordered pairs i < j."""
    features = torch.nn.functional.normalize(features.double(), dim=-1)
    return torch.pdist(features, p=2).pow(2).mul(-t).exp().mean().log().item()


def test_spread_out_matches_bruteforce():
    torch.manual_seed(0)
    features = torch.randn(53, 16) + 0.7  # shifted so the mean similarity is clearly non-zero
    assert spread_out_metric(features) == pytest.approx(_bruteforce_spread_out(features), abs=1e-10)


def test_spread_out_extremes():
    # Identical directions (any scale) form a degenerate cone; antipodal pairs average to -1.
    assert spread_out_metric(torch.tensor([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]])) == pytest.approx(1.0)
    assert spread_out_metric(torch.tensor([[1.0, 0.0], [-1.0, 0.0]])) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="at least two"):
        spread_out_metric(torch.ones(1, 4))


@pytest.mark.parametrize("t", [0.5, 2.0, 50.0])
@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_uniformity_matches_reference(t, chunk_size):
    torch.manual_seed(0)
    features = torch.randn(53, 16) + 0.7
    expected = _reference_uniformity(features, t)
    assert uniformity_metric(features, t=t, chunk_size=chunk_size) == pytest.approx(expected, abs=1e-10)


def test_uniformity_extremes():
    # Collapsed (any scale) -> 0; antipodal unit vectors are at squared distance 4 -> -4t.
    assert uniformity_metric(torch.tensor([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]])) == pytest.approx(0.0, abs=1e-12)
    assert uniformity_metric(torch.tensor([[3.0, 0.0], [-1.0, 0.0]]), t=2.0) == pytest.approx(-8.0)
    # Stays finite where a naive exp() would underflow to log(0).
    assert uniformity_metric(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), t=1000.0) == pytest.approx(-4000.0)
    # Spread uniformly in dimension d: cos(e_i, e_j) has variance ~1/d, so the value is about -2t + 2t^2/d.
    torch.manual_seed(0)
    assert uniformity_metric(torch.randn(2000, 512), t=2.0) == pytest.approx(-4.0 + 8.0 / 512, abs=2e-3)
    with pytest.raises(ValueError, match="at least two"):
        uniformity_metric(torch.ones(1, 4))
    with pytest.raises(ValueError, match="positive"):
        uniformity_metric(torch.randn(3, 4), t=0.0)


def test_cone_eval_end_to_end(tmp_path):
    torch.manual_seed(0)
    captions_per_image = [5, 5, 6, 1, 0, 5, 3]
    shard_dir = _write_shards(tmp_path / "flickr30k", captions_per_image)
    args = _args(cone_val=[str(shard_dir)], zeroshot_frequency=2, epochs=3, cone_uniformity_t=3.0)
    model = create_tiny_model().eval()
    task = CLIPTask(model, rank=0, world_size=1).eval()
    data = build_cone_datasets(args, transforms.ToTensor())
    assert list(data) == ["cone-flickr30k"]

    assert cone_eval(task, data, epoch=1, args=args, tokenizer=_tokenizer) == {}
    metrics = zero_shot_eval_all(task, data, epoch=2, args=args, tokenizer=_tokenizer)
    assert metrics.keys() == {
        f"flickr30k-cone-{modality}_{name}" for modality in ("image", "text") for name in ("spread_out", "uniformity")
    }

    texts = [f"{i}x{c}y{i * c}" for i, n in enumerate(captions_per_image) for c in range(n)]
    with torch.no_grad():
        text_features = model.encode_text(_tokenizer(texts))
    assert metrics["flickr30k-cone-text_spread_out"] == pytest.approx(_bruteforce_spread_out(text_features), abs=1e-5)
    assert metrics["flickr30k-cone-text_uniformity"] == pytest.approx(
        _reference_uniformity(text_features, t=3.0), abs=1e-5)
    assert -1.0 <= metrics["flickr30k-cone-image_spread_out"] <= 1.0
    assert -12.0 <= metrics["flickr30k-cone-image_uniformity"] <= 0.0


def test_cone_eval_defaults_to_paper_temperature(tmp_path):
    shard_dir = _write_shards(tmp_path / "flickr30k", [2, 3])
    args = _args(cone_val=[str(shard_dir)])
    assert not hasattr(args, "cone_uniformity_t")
    model = create_tiny_model().eval()
    task = CLIPTask(model, rank=0, world_size=1).eval()
    metrics = cone_eval(task, build_cone_datasets(args, transforms.ToTensor()), 1, args, tokenizer=_tokenizer)
    texts = [f"{i}x{c}y{i * c}" for i, n in enumerate([2, 3]) for c in range(n)]
    with torch.no_grad():
        expected = _reference_uniformity(model.encode_text(_tokenizer(texts)), t=2.0)
    assert math.isfinite(metrics["flickr30k-cone-text_uniformity"])
    assert metrics["flickr30k-cone-text_uniformity"] == pytest.approx(expected, abs=1e-5)


def test_cone_dataset_errors():
    with pytest.raises(ValueError, match="Duplicate --cone-val"):
        build_cone_datasets(_args(cone_val=["/a/x/test", "/b/x/test"]), None)
    with pytest.raises(ValueError, match="NaFlex"):
        build_cone_datasets(_args(cone_val=["x=/tmp"]), None, naflex_data_config=object())

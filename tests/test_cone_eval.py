"""Cone-effect eval (--cone-val)."""
import pytest
import torch
from torchvision import transforms

from open_clip.task import CLIPTask
from open_clip_train.cone_eval import build_cone_datasets, cone_eval, mean_pairwise_cosine
from open_clip_train.train import zero_shot_eval_all
from test_retrieval_eval import _args, _tokenizer, _write_shards
from util_test import create_tiny_model


def _bruteforce_mean_cosine(features):
    features = torch.nn.functional.normalize(features.double(), dim=-1)
    sims = features @ features.T
    off_diagonal = ~torch.eye(len(features), dtype=torch.bool)
    return sims[off_diagonal].mean().item()


def test_mean_pairwise_cosine_matches_bruteforce():
    torch.manual_seed(0)
    features = torch.randn(53, 16) + 0.7  # shifted so the mean similarity is clearly non-zero
    assert mean_pairwise_cosine(features) == pytest.approx(_bruteforce_mean_cosine(features), abs=1e-10)


def test_mean_pairwise_cosine_extremes():
    # Identical directions (any scale) form a degenerate cone; antipodal pairs average to -1.
    assert mean_pairwise_cosine(torch.tensor([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]])) == pytest.approx(1.0)
    assert mean_pairwise_cosine(torch.tensor([[1.0, 0.0], [-1.0, 0.0]])) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="at least two"):
        mean_pairwise_cosine(torch.ones(1, 4))


def test_cone_eval_end_to_end(tmp_path):
    torch.manual_seed(0)
    captions_per_image = [5, 5, 6, 1, 0, 5, 3]
    shard_dir = _write_shards(tmp_path / "flickr30k", captions_per_image)
    args = _args(cone_val=[str(shard_dir)], zeroshot_frequency=2, epochs=3)
    model = create_tiny_model().eval()
    task = CLIPTask(model, rank=0, world_size=1).eval()
    data = build_cone_datasets(args, transforms.ToTensor())
    assert list(data) == ["cone-flickr30k"]

    assert cone_eval(task, data, epoch=1, args=args, tokenizer=_tokenizer) == {}
    metrics = zero_shot_eval_all(task, data, epoch=2, args=args, tokenizer=_tokenizer)
    assert metrics.keys() == {"flickr30k-cone-image_mean_cosine", "flickr30k-cone-text_mean_cosine"}

    texts = [f"{i}x{c}y{i * c}" for i, n in enumerate(captions_per_image) for c in range(n)]
    with torch.no_grad():
        expected_text = _bruteforce_mean_cosine(model.encode_text(_tokenizer(texts)))
    assert metrics["flickr30k-cone-text_mean_cosine"] == pytest.approx(expected_text, abs=1e-5)
    assert -1.0 <= metrics["flickr30k-cone-image_mean_cosine"] <= 1.0


def test_cone_dataset_errors():
    with pytest.raises(ValueError, match="Duplicate --cone-val"):
        build_cone_datasets(_args(cone_val=["/a/x/test", "/b/x/test"]), None)
    with pytest.raises(ValueError, match="NaFlex"):
        build_cone_datasets(_args(cone_val=["x=/tmp"]), None, naflex_data_config=object())

"""Multi-caption zero-shot retrieval eval (--retrieval-val)."""
import io
import tarfile
import types

import pytest
import torch
from PIL import Image
from torchvision import transforms

from open_clip.task import CLIPTask
from open_clip_train.retrieval_eval import (
    RETRIEVAL_RECALL_KS,
    build_retrieval_datasets,
    compute_retrieval_metrics,
    parse_retrieval_specs,
    resolve_retrieval_shards,
    retrieval_eval,
    split_captions,
)
from open_clip_train.train import zero_shot_eval_all
from util_test import create_tiny_model


def _reference_metrics(image_features, text_features, text_to_image, ks=RETRIEVAL_RECALL_KS):
    """Brute-force multi-caption recall: full sort per query."""
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    scores = image_features @ text_features.T
    metrics = {}
    for k in ks:
        i2t_topk = scores.topk(min(k, scores.shape[1]), dim=1).indices
        i2t_hits = [
            any(text_to_image[j].item() == i for j in i2t_topk[i]) for i in range(scores.shape[0])
        ]
        t2i_topk = scores.T.topk(min(k, scores.shape[0]), dim=1).indices
        t2i_hits = [text_to_image[j].item() in t2i_topk[j].tolist() for j in range(scores.shape[1])]
        metrics[f"image_to_text_R@{k}"] = sum(i2t_hits) / len(i2t_hits)
        metrics[f"text_to_image_R@{k}"] = sum(t2i_hits) / len(t2i_hits)
    return metrics


def test_compute_retrieval_metrics_matches_bruteforce():
    torch.manual_seed(0)
    num_images = 37
    captions_per_image = torch.randint(1, 7, (num_images,))
    text_to_image = torch.repeat_interleave(torch.arange(num_images), captions_per_image)
    image_features = torch.randn(num_images, 16)
    # Captions are noisy copies of their image so recalls are neither 0 nor 1.
    text_features = image_features[text_to_image] + 1.5 * torch.randn(len(text_to_image), 16)

    expected = _reference_metrics(image_features, text_features, text_to_image)
    for chunk_size in (1, 5, 1024):
        actual = compute_retrieval_metrics(image_features, text_features, text_to_image, chunk_size=chunk_size)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value), (key, chunk_size)
    assert 0 < expected["image_to_text_R@1"] < 1
    assert 0 < expected["text_to_image_R@1"] < 1


def test_image_to_text_hits_on_any_caption():
    image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Image 0 owns captions 0 and 1. Caption 0 ranks last for image 0, but caption 1 ranks first, so image 0 is
    # a hit at R@1. Image 1's only caption (2) is outranked by caption 0, so image 1 is a miss.
    text_features = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
    text_to_image = torch.tensor([0, 0, 1])
    metrics = compute_retrieval_metrics(image_features, text_features, text_to_image, recall_ks=(1, 5))
    assert metrics["image_to_text_R@1"] == 0.5
    assert metrics["image_to_text_R@5"] == 1.0
    # Caption 0 retrieves image 1 (miss); captions 1 and 2 retrieve their own images.
    assert metrics["text_to_image_R@1"] == pytest.approx(2 / 3)


def test_parse_retrieval_specs():
    assert parse_retrieval_specs(["coco=/data/mscoco/test", "/data/flickr30k/test", "/data/x/{0..4}.tar"]) == [
        ("coco", "/data/mscoco/test"),
        ("flickr30k", "/data/flickr30k/test"),
        ("x", "/data/x/{0..4}.tar"),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_retrieval_specs(["/a/flickr30k/test", "/b/flickr30k/val"])
    with pytest.raises(ValueError, match="Invalid"):
        parse_retrieval_specs(["name="])


def test_split_captions():
    assert split_captions(b"A dog.\n\n  A cat. \nA bird.") == ["A dog.", "A cat.", "A bird."]


def _write_shards(root, captions_per_image, images_per_shard=3, image_size=32):
    root.mkdir()
    shard_idx = 0
    for start in range(0, len(captions_per_image), images_per_shard):
        with tarfile.open(root / f"{shard_idx}.tar", "w") as tar:
            for i in range(start, min(start + images_per_shard, len(captions_per_image))):
                color = ((37 * i) % 256, (91 * i) % 256, (151 * i) % 256)
                image_bytes = io.BytesIO()
                Image.new("RGB", (image_size, image_size), color).save(image_bytes, format="PNG")
                caption_text = "\n".join(f"{i}x{c}y{i * c}" for c in range(captions_per_image[i]))
                for name, payload in ((f"{i:05d}.png", image_bytes.getvalue()),
                                      (f"{i:05d}.txt", caption_text.encode())):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
        shard_idx += 1
    return root


def _tokenizer(texts, context_length=8, vocab_size=64):
    tokens = torch.zeros(len(texts), context_length, dtype=torch.long)
    for row, text in enumerate(texts):
        ids = [(ord(ch) * 7 + pos) % (vocab_size - 1) + 1 for pos, ch in enumerate(text)][:context_length]
        tokens[row, :len(ids)] = torch.tensor(ids)
    return tokens


def _args(**overrides):
    args = types.SimpleNamespace(
        device="cpu", precision="fp32", rank=0, world_size=1, distributed=False, fsdp=False,
        batch_size=4, workers=0, zeroshot_frequency=1, epochs=2, model="tiny-clip",
    )
    args.__dict__.update(overrides)
    return args


@pytest.mark.parametrize("workers", [0, 2])
def test_retrieval_eval_end_to_end(tmp_path, workers):
    torch.manual_seed(0)
    captions_per_image = [5, 5, 6, 1, 0, 5, 3]  # the caption-less image is skipped
    shard_dir = _write_shards(tmp_path / "test", captions_per_image)
    preprocess = transforms.ToTensor()
    args = _args(workers=workers, retrieval_val=[f"tiny={shard_dir}"])

    model = create_tiny_model().eval()
    task = CLIPTask(model, rank=0, world_size=1).eval()
    data = build_retrieval_datasets(args, preprocess)
    assert list(data) == ["retrieval-tiny"]

    metrics = zero_shot_eval_all(task, data, epoch=1, args=args, tokenizer=_tokenizer)

    # Recompute directly from the encoders, independent of the dataloader and chunked scoring.
    images, texts, owners = [], [], []
    for i, n in enumerate(captions_per_image):
        if n == 0:
            continue
        image_path = sorted(shard_dir.glob("*.tar"))[i // 3]
        with tarfile.open(image_path) as tar:
            images.append(preprocess(Image.open(tar.extractfile(f"{i:05d}.png")).convert("RGB")))
        texts.extend(f"{i}x{c}y{i * c}" for c in range(n))
        owners.extend([len(images) - 1] * n)
    with torch.no_grad():
        expected = _reference_metrics(
            model.encode_image(torch.stack(images)), model.encode_text(_tokenizer(texts)), torch.tensor(owners),
        )
    assert metrics.keys() == {f"tiny-retrieval-{key}" for key in expected}
    for key, value in expected.items():
        assert metrics[f"tiny-retrieval-{key}"] == pytest.approx(value), key


def test_retrieval_eval_follows_zeroshot_frequency(tmp_path):
    shard_dir = _write_shards(tmp_path / "flickr30k", [2, 2])
    args = _args(retrieval_val=[str(shard_dir)], zeroshot_frequency=2, epochs=3)
    task = CLIPTask(create_tiny_model(), rank=0, world_size=1).eval()
    data = build_retrieval_datasets(args, transforms.ToTensor())
    assert list(data) == ["retrieval-flickr30k"]

    assert retrieval_eval(task, data, epoch=1, args=args, tokenizer=_tokenizer) == {}
    assert "flickr30k-retrieval-image_to_text_R@1" in retrieval_eval(task, data, 2, args, tokenizer=_tokenizer)
    assert "flickr30k-retrieval-text_to_image_R@1" in retrieval_eval(task, data, 3, args, tokenizer=_tokenizer)
    args.zeroshot_frequency = 0
    assert retrieval_eval(task, data, epoch=2, args=args, tokenizer=_tokenizer) == {}


def test_retrieval_dataset_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No .tar shards"):
        resolve_retrieval_shards(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="NaFlex"):
        build_retrieval_datasets(_args(retrieval_val=["x=/tmp"]), None, naflex_data_config=object())

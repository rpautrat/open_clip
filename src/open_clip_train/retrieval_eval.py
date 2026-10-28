"""Zero-shot image-text retrieval eval on multi-caption benchmarks (e.g. Flickr30k, MSCOCO 5k).

Datasets are webdataset ``.tar`` shards where each sample holds one image plus a ``txt`` member with one caption
per line (the layout of CLIP_benchmark's ``wds_flickr30k`` / ``wds_mscoco_captions``). An image-to-text query is a
hit at K if *any* of the image's captions ranks in the top K, matching the standard benchmark protocol.
"""
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader

from open_clip import get_input_dtype, get_tokenizer
from open_clip.task import get_model_from_task
from open_clip.utils import move_to_device
from open_clip_train.eval_utils import iter_eval_batches
from open_clip_train.precision import get_autocast

_logger = logging.getLogger(__name__)

RETRIEVAL_DATA_PREFIX = "retrieval-"
RETRIEVAL_IMAGE_KEYS = "jpg;jpeg;png;webp"
RETRIEVAL_RECALL_KS = (1, 5)
_SPLIT_DIR_NAMES = {"test", "val", "valid", "validation"}


@dataclass
class RetrievalData:
    dataloader: Optional[DataLoader]
    name: str


def is_retrieval_compatible(model_or_task) -> bool:
    model = get_model_from_task(model_or_task)
    return hasattr(model, "encode_image") and hasattr(model, "encode_text")


def validate_retrieval_compatible(model_or_task):
    if not is_retrieval_compatible(model_or_task):
        raise ValueError("Zero-shot retrieval evaluation requires a model with image and text encoders.")


def _default_retrieval_name(path: str) -> str:
    path = path.rstrip("/")
    if path.endswith(".tar") or "{" in path:
        path = os.path.dirname(path)
    name = os.path.basename(path)
    if name in _SPLIT_DIR_NAMES:
        name = os.path.basename(os.path.dirname(path))
    return name


def parse_retrieval_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse ``name=path`` (or bare ``path``) entries into unique ``(name, path)`` pairs."""
    parsed = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = _default_retrieval_name(spec), spec
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --retrieval-val entry {spec!r}; expected 'name=path' or 'path'.")
        parsed.append((name, path))
    names = [name for name, _ in parsed]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate --retrieval-val names {duplicates}; use 'name=path' to disambiguate.")
    return parsed


def _natural_key(path: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def resolve_retrieval_shards(path: str) -> List[str]:
    """A directory expands to its ``*.tar`` files; anything else is a (brace-expandable) shard spec."""
    if os.path.isdir(path):
        shards = sorted(glob.glob(os.path.join(path, "*.tar")), key=_natural_key)
        if not shards:
            raise ValueError(f"No .tar shards found in retrieval dataset directory {path!r}.")
        return shards
    return list(wds.shardlists.expand_urls(path))


def split_captions(text) -> List[str]:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _collate_retrieval(samples):
    images, captions = zip(*samples)
    return torch.stack(images), list(captions)


def get_retrieval_dataset(path: str, name: str, preprocess_fn, batch_size: int, workers: int) -> RetrievalData:
    shards = resolve_retrieval_shards(path)
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.reraise_exception),
        wds.decode("pilrgb", handler=wds.reraise_exception),
        wds.to_tuple(RETRIEVAL_IMAGE_KEYS, "txt", handler=wds.reraise_exception),
        wds.map_tuple(preprocess_fn, split_captions),
        # Images without any caption cannot be scored as image-to-text queries.
        wds.select(lambda sample: len(sample[1]) > 0),
        wds.batched(batch_size, collation_fn=_collate_retrieval, partial=True),
    )
    num_workers = min(workers, len(shards))
    dataloader = DataLoader(pipeline, batch_size=None, shuffle=False, num_workers=num_workers)
    return RetrievalData(dataloader=dataloader, name=name)


def build_retrieval_datasets(args, preprocess_val, naflex_data_config=None) -> Dict[str, RetrievalData]:
    specs = getattr(args, "retrieval_val", None)
    if not specs:
        return {}
    if naflex_data_config is not None:
        raise ValueError("--retrieval-val does not support NaFlex models yet.")
    return {
        f"{RETRIEVAL_DATA_PREFIX}{name}": get_retrieval_dataset(path, name, preprocess_val, args.batch_size, args.workers)
        for name, path in parse_retrieval_specs(specs)
    }


def _extract_features(output, key: str, tuple_index: int):
    if isinstance(output, dict):
        return output[f"{key}_features"]
    if isinstance(output, (tuple, list)):
        return output[tuple_index]
    return output


def _prepare_images(model_or_task, images, device, input_dtype):
    if hasattr(model_or_task, "prepare_batch"):
        return model_or_task.prepare_batch({"image": images}, device, input_dtype)["image"]
    return move_to_device(images, device, input_dtype)


def _create_dummy_inputs(model_or_task, device, input_dtype):
    if hasattr(model_or_task, "create_dummy_batch"):
        batch = model_or_task.create_dummy_batch(batch_size=1, device=device, dtype=input_dtype)
        return batch["image"], batch["text"]
    model = get_model_from_task(model_or_task)
    image_size = model.visual.image_size
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    image = torch.zeros(1, 3, *image_size, device=device, dtype=input_dtype)
    text = torch.zeros(1, model.context_length, device=device, dtype=torch.long)
    return image, text


@torch.no_grad()
def compute_retrieval_metrics(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        text_to_image: torch.Tensor,
        recall_ks: Sequence[int] = RETRIEVAL_RECALL_KS,
        chunk_size: int = 1024,
) -> Dict[str, float]:
    """Multi-caption R@K. ``text_to_image[j]`` is the index of the image paired with caption ``j``.

    Ranks count strictly higher-scoring candidates, so exact score ties resolve in the query's favor.
    """
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    text_to_image = text_to_image.to(image_features.device)
    num_images, num_texts = image_features.shape[0], text_features.shape[0]

    text_ranks = torch.empty(num_texts, dtype=torch.long, device=image_features.device)
    for start in range(0, num_texts, chunk_size):
        end = min(start + chunk_size, num_texts)
        scores = text_features[start:end] @ image_features.T
        target = scores.gather(1, text_to_image[start:end, None])
        text_ranks[start:end] = (scores > target).sum(dim=1)

    image_ranks = torch.empty(num_images, dtype=torch.long, device=image_features.device)
    for start in range(0, num_images, chunk_size):
        end = min(start + chunk_size, num_images)
        scores = image_features[start:end] @ text_features.T
        image_idx = torch.arange(start, end, device=scores.device)[:, None]
        best_positive = scores.masked_fill(text_to_image[None, :] != image_idx, float("-inf")).amax(dim=1)
        image_ranks[start:end] = (scores > best_positive[:, None]).sum(dim=1)

    metrics = {}
    for k in recall_ks:
        metrics[f"image_to_text_R@{k}"] = (image_ranks < k).float().mean().item()
        metrics[f"text_to_image_R@{k}"] = (text_ranks < k).float().mean().item()
    return metrics


def run_retrieval_eval(model_or_task, dataloader, args, tokenizer, use_fsdp_eval=False) -> Dict[str, float]:
    """Encode all images/captions, then score on rank 0. Under FSDP every rank runs matching forwards."""
    device = torch.device(args.device)
    input_dtype = get_input_dtype(args.precision)
    autocast = get_autocast(args.precision, device_type=device.type, fsdp=getattr(args, "fsdp", False))
    score = not use_fsdp_eval or args.rank == 0
    dummy_image, dummy_text = (None, None) if score else _create_dummy_inputs(model_or_task, device, input_dtype)

    image_features, text_features, text_to_image = [], [], []
    num_images = 0
    with torch.inference_mode():
        for batch in iter_eval_batches(dataloader, device, args.rank, use_fsdp_eval, args.batch_size):
            if score:
                images, captions = batch
                images = _prepare_images(model_or_task, images, device, input_dtype)
                texts = tokenizer([caption for caps in captions for caption in caps]).to(device)
                owners = [num_images + i for i, caps in enumerate(captions) for _ in caps]
            else:
                images, texts = dummy_image, dummy_text
            with autocast():
                image_out = _extract_features(model_or_task(image=images), "image", 0)
                text_out = _extract_features(model_or_task(text=texts), "text", 1)
            if score:
                image_features.append(image_out.float())
                text_features.append(text_out.float())
                text_to_image.append(torch.tensor(owners, dtype=torch.long))
                num_images += image_out.shape[0]

    if not score or num_images == 0:
        return {}
    return compute_retrieval_metrics(
        torch.cat(image_features), torch.cat(text_features), torch.cat(text_to_image),
    )


def retrieval_eval(model_or_task, data, epoch, args, tokenizer=None) -> Dict[str, float]:
    """Run every ``retrieval-<name>`` entry in ``data`` on the ``--zeroshot-frequency`` schedule."""
    keys = [key for key in data if key.startswith(RETRIEVAL_DATA_PREFIX)]
    if not keys:
        return {}
    validate_retrieval_compatible(model_or_task)
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    use_fsdp_eval = getattr(args, "fsdp", False) and getattr(args, "distributed", False)
    is_rank0 = args.rank == 0
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    results = {}
    for key in keys:
        retrieval_data = data[key]
        if is_rank0:
            _logger.info(f"Starting zero-shot retrieval on {retrieval_data.name}.")
        metrics = run_retrieval_eval(
            model_or_task,
            retrieval_data.dataloader if is_rank0 else None,
            args,
            tokenizer,
            use_fsdp_eval=use_fsdp_eval,
        )
        if is_rank0:
            results.update({f"{retrieval_data.name}-retrieval-{name}": value for name, value in metrics.items()})
    if is_rank0:
        _logger.info("Finished zero-shot retrieval.")
    return results

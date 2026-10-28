"""Cone-effect eval: mean pairwise cosine similarity within the image and within the text embedding sets.

A value near 0 means embeddings spread over the sphere; a value near 1 means they collapse into a narrow cone
(the "cone effect" of Liang et al., 2022, "Mind the Gap"). Reads the same multi-caption webdataset shards as
``--retrieval-val`` (e.g. the Flickr30k test split: 1,000 images / 5,000 captions).
"""
import logging
from typing import Dict

import torch
import torch.nn.functional as F

from open_clip import get_tokenizer
from open_clip_train.retrieval_eval import (
    RetrievalData,
    encode_image_text_dataset,
    get_retrieval_dataset,
    parse_retrieval_specs,
    validate_retrieval_compatible,
)

_logger = logging.getLogger(__name__)

CONE_DATA_PREFIX = "cone-"


@torch.no_grad()
def mean_pairwise_cosine(features: torch.Tensor) -> float:
    """Mean cosine similarity over all distinct pairs ``i != j``.

    For N unit vectors, ``mean_{i != j} cos(f_i, f_j) = (||sum_i f_i||^2 - N) / (N * (N - 1))``: the squared norm
    of the sum covers all ordered pairs, ``- N`` removes the N self-pairs (each cos = 1), and ``N * (N - 1)`` is
    the number of distinct ordered pairs. This is exact and O(N * D).
    """
    num = features.shape[0]
    if num < 2:
        raise ValueError("Mean pairwise cosine similarity needs at least two embeddings.")
    features = F.normalize(features.double(), dim=-1)
    total = features.sum(dim=0).square().sum()
    return ((total - num) / (num * (num - 1))).item()


def build_cone_datasets(args, preprocess_val, naflex_data_config=None) -> Dict[str, RetrievalData]:
    specs = getattr(args, "cone_val", None)
    if not specs:
        return {}
    if naflex_data_config is not None:
        raise ValueError("--cone-val does not support NaFlex models yet.")
    return {
        f"{CONE_DATA_PREFIX}{name}": get_retrieval_dataset(path, name, preprocess_val, args.batch_size, args.workers)
        for name, path in parse_retrieval_specs(specs, option="--cone-val")
    }


def run_cone_eval(model_or_task, dataloader, args, tokenizer, use_fsdp_eval=False) -> Dict[str, float]:
    encoded = encode_image_text_dataset(model_or_task, dataloader, args, tokenizer, use_fsdp_eval=use_fsdp_eval)
    if encoded is None:
        return {}
    image_features, text_features, _ = encoded
    return {
        "image_mean_cosine": mean_pairwise_cosine(image_features),
        "text_mean_cosine": mean_pairwise_cosine(text_features),
    }


def cone_eval(model_or_task, data, epoch, args, tokenizer=None) -> Dict[str, float]:
    """Run every ``cone-<name>`` entry in ``data`` on the ``--zeroshot-frequency`` schedule."""
    keys = [key for key in data if key.startswith(CONE_DATA_PREFIX)]
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
        cone_data = data[key]
        if is_rank0:
            _logger.info(f"Starting cone-effect eval on {cone_data.name}.")
        metrics = run_cone_eval(
            model_or_task,
            cone_data.dataloader if is_rank0 else None,
            args,
            tokenizer,
            use_fsdp_eval=use_fsdp_eval,
        )
        if is_rank0:
            results.update({f"{cone_data.name}-cone-{name}": value for name, value in metrics.items()})
    return results

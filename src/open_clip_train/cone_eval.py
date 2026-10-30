"""Cone-effect eval: how much the image embeddings, and separately the text embeddings, collapse into a narrow cone
(the "cone effect" of Liang et al., 2022, "Mind the Gap"). Two metrics are reported per modality:

- spread-out: mean pairwise cosine similarity. ~0 for embeddings spread uniformly over the sphere, 1 for a fully
  collapsed set.
- uniformity: ``log mean_{i != j} exp(-t * ||e_i - e_j||^2)`` on L2-normalized embeddings (Wang & Isola, ICML 2020,
  "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere").
  Ranges over ``[-4t, 0]``: 0 for a fully collapsed set, lower is more uniform (about ``-2t + 2t^2/d`` for
  embeddings spread uniformly over the sphere in dimension ``d``).

Reads the same multi-caption webdataset shards as ``--retrieval-val`` (e.g. the Flickr30k test split: 1,000 images /
5,000 captions).
"""
import logging
import math
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
DEFAULT_UNIFORMITY_T = 2.0


@torch.no_grad()
def spread_out_metric(features: torch.Tensor) -> float:
    """Spread-out metric: mean cosine similarity over all distinct pairs ``i != j``.

    For N unit vectors, ``mean_{i != j} cos(f_i, f_j) = (||sum_i f_i||^2 - N) / (N * (N - 1))``: the squared norm
    of the sum covers all ordered pairs, ``- N`` removes the N self-pairs (each cos = 1), and ``N * (N - 1)`` is
    the number of distinct ordered pairs. This is exact and O(N * D).
    """
    num = features.shape[0]
    if num < 2:
        raise ValueError("The spread-out metric needs at least two embeddings.")
    features = F.normalize(features.double(), dim=-1)
    total = features.sum(dim=0).square().sum()
    return ((total - num) / (num * (num - 1))).item()


@torch.no_grad()
def uniformity_metric(features: torch.Tensor, t: float = DEFAULT_UNIFORMITY_T, chunk_size: int = 1024) -> float:
    """Uniformity metric: ``log mean_{i != j} exp(-t * ||e_i - e_j||^2)`` on L2-normalized embeddings.

    The mean over ordered pairs ``i != j`` equals the paper's mean over unordered pairs ``i < j``. For unit vectors
    ``||e_i - e_j||^2 = 2 - 2 * cos(e_i, e_j)``. Computed in float64 as a chunked log-sum-exp, so it is stable for
    any ``t`` and needs O(chunk_size * N) memory.
    """
    num = features.shape[0]
    if num < 2:
        raise ValueError("The uniformity metric needs at least two embeddings.")
    if t <= 0:
        raise ValueError(f"The uniformity temperature t must be positive, got {t}.")
    features = F.normalize(features.double(), dim=-1)
    chunk_lse = []
    for start in range(0, num, chunk_size):
        end = min(start + chunk_size, num)
        sq_dist = (2 - 2 * features[start:end] @ features.T).clamp_min(0)
        logits = -t * sq_dist
        rows = torch.arange(end - start, device=logits.device)
        logits[rows, rows + start] = float("-inf")
        chunk_lse.append(torch.logsumexp(logits.flatten(), dim=0))
    total_lse = torch.logsumexp(torch.stack(chunk_lse), dim=0)
    return (total_lse - math.log(num * (num - 1))).item()


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


def compute_cone_metrics(features: torch.Tensor, t: float = DEFAULT_UNIFORMITY_T) -> Dict[str, float]:
    return {"spread_out": spread_out_metric(features), "uniformity": uniformity_metric(features, t=t)}


def run_cone_eval(model_or_task, dataloader, args, tokenizer, use_fsdp_eval=False) -> Dict[str, float]:
    encoded = encode_image_text_dataset(model_or_task, dataloader, args, tokenizer, use_fsdp_eval=use_fsdp_eval)
    if encoded is None:
        return {}
    image_features, text_features, _ = encoded
    t = getattr(args, "cone_uniformity_t", DEFAULT_UNIFORMITY_T)
    metrics = {}
    for modality, features in (("image", image_features), ("text", text_features)):
        metrics.update({f"{modality}_{name}": value for name, value in compute_cone_metrics(features, t=t).items()})
    return metrics


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

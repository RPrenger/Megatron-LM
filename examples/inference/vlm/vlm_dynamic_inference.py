# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# pylint: disable=bad-builtin

"""VLM dynamic inference example.

Uses the unified DynamicInferenceEngine for BOTH text-only (GPT) AND
multimodal (VLM) models/requests.  The model type is auto-detected from the
checkpoint:
  - GPT checkpoint  -> loads GPT model, runs text-only inference
  - VLM checkpoint  -> loads LLaVA model, supports text + image requests

Usage modes:
  - Text-only:   engine.add_request(id, prompt, sampling_params)
  - Multimodal:  engine.add_request(id, prompt, sampling_params,
                     imgs=..., num_tiles=..., num_img_embeddings_per_tile=...)
                 or engine.add_request(id, prompt, sampling_params,
                     imgs=..., imgs_sizes=...)  # dynamic resolution

V1 limitations:
    - PP=1 only (vision encoder must be on same rank).
    - No chunked prefill for VLM requests.
"""

import json
import math
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir,
            "multimodal",
        )
    )
)

from examples.inference.gpt.utils import (
    Request,
    build_requests,
    get_curr_time,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.module import MegatronModule
from megatron.inference.utils import add_inference_args, get_inference_config_from_model_and_args
from megatron.training import get_args, get_model as _get_model
from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint
from megatron.training.initialize import initialize_megatron


def add_vlm_inference_args(parser):
    """Add VLM-specific inference arguments on top of the standard inference args."""
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="VLM dynamic inference")
    group.add_argument(
        "--input-image-path",
        type=str,
        default=None,
        help="Path to input image(s). Can be a single image or directory.",
    )
    group.add_argument(
        "--input-prompts-json",
        type=str,
        default=None,
        help="Path to JSON file with prompts and image paths. "
        'Format: [{"prompt": "...", "image": "path/to/image.jpg"}, ...]',
    )
    return parser


# ---------------------------------------------------------------------------
# Model loading with auto-detection
# ---------------------------------------------------------------------------

def _detect_vlm_from_checkpoint(args):
    """Peek at the checkpoint's saved training args to detect VLM vs GPT.

    Returns True if the checkpoint was trained as a VLM (has language_model_type),
    False otherwise.  As a side-effect, copies VLM-specific args from the
    checkpoint into the current args namespace so that the multimodal
    model_provider can access them.
    """
    result = load_args_from_checkpoint(args)
    if not isinstance(result, tuple):
        # Checkpoint not found or has no args — assume GPT.
        return False

    _, checkpoint_args = result
    if not hasattr(checkpoint_args, 'language_model_type'):
        return False
    if checkpoint_args.language_model_type is None:
        return False

    # Copy VLM-specific attrs that aren't in the standard checkpoint whitelist.
    vlm_attrs = [
        'language_model_type', 'vision_model_type', 'decoder_seq_length',
        'use_te', 'disable_vision_class_token', 'pixel_shuffle',
        'use_tile_tags', 'max_num_tiles', 'use_thumbnail', 'use_tiling',
        'tokenizer_prompt_format', 'recompute_vision', 'num_frames',
        'freeze_LM', 'freeze_ViT', 'allow_missing_vision_projection_checkpoint',
        'pixel_mean', 'pixel_std', 'use_area_weighted_aspect_ratio',
        # Dynamic resolution args
        'dynamic_resolution', 'dynamic_resolution_min_patches',
        'dynamic_resolution_max_patches',
        'class_token_len',
        'radio_force_cpe_eval_mode', 'radio_force_eval_mode',
        'radio_interpolate_only_cpe', 'radio_cpe_aspect_ratio_select',
        'radio_disable_cpe',
    ]
    for attr in vlm_attrs:
        val = getattr(checkpoint_args, attr, None)
        if val is not None and not hasattr(args, attr):
            setattr(args, attr, val)
        elif val is not None:
            # Override only if current value is None/default
            if getattr(args, attr, None) is None:
                setattr(args, attr, val)

    return True


def get_model(is_vlm: bool) -> MegatronModule:
    """Build and load the model.  Dispatches to the right model_provider."""
    args = get_args()

    if is_vlm:
        # Import model_provider from examples/multimodal/.
        from model import model_provider
        model = _get_model(partial(model_provider), wrap_with_ddp=False)
    else:
        # GPT model — same path as gpt_dynamic_inference.py.
        from gpt_builders import gpt_builder
        from model_provider import model_provider
        model = _get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)

    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not args.inference_ckpt_non_strict,
    )

    assert len(model) == 1, "Virtual PP not supported for VLM inference"
    model = model[0]
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image preprocessing (only used for VLM + multimodal mode)
# ---------------------------------------------------------------------------

# Pixel statistics for different vision models.
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
_PIXEL_STATS = {
    "radio": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "radio-so400m": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "radio-g": ([0.4850, 0.4560, 0.4060], [0.2230, 0.2240, 0.2250]),
    "cradio-g": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "clip": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "siglip": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "internvit": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def dynamic_res_preprocess(
    image, min_patches=1, max_patches=128, res_step=16, factor_max=1.0, pixel_shuffle=False,
):
    """Resize image to fit within [min_patches, max_patches] preserving aspect ratio.

    Based on megatron-lm's dynamic_res_preprocess.  For pixel_shuffle, patch grid
    dimensions are rounded to even numbers for compatibility.
    """
    orig_width, orig_height = image.size

    closest_patch_height = round(orig_height / res_step + 0.5)
    closest_patch_width = round(orig_width / res_step + 0.5)
    patches = closest_patch_height * closest_patch_width

    factor = min(math.sqrt(max_patches / patches), factor_max)
    target_patch_height = math.floor(factor * closest_patch_height)
    target_patch_width = math.floor(factor * closest_patch_width)

    if target_patch_height * target_patch_width < min_patches:
        up_factor = math.sqrt(min_patches / (target_patch_height * target_patch_width))
        target_patch_height = math.ceil(up_factor * target_patch_height)
        target_patch_width = math.ceil(up_factor * target_patch_width)

    # Round patch grid to be divisible by 2 for pixel shuffle compatibility.
    if pixel_shuffle:
        rem_h = target_patch_height % 2
        if rem_h != 0:
            if (target_patch_height + 1) * target_patch_width <= max_patches:
                target_patch_height += 1
            else:
                target_patch_height = max(1, target_patch_height - 1)

        rem_w = target_patch_width % 2
        if rem_w != 0:
            if target_patch_height * (target_patch_width + 1) <= max_patches:
                target_patch_width += 1
            else:
                target_patch_width = max(1, target_patch_width - 1)

    assert target_patch_height * target_patch_width <= max_patches

    resized_img = image.resize((target_patch_width * res_step, target_patch_height * res_step))
    return resized_img


def load_and_preprocess_image_dynamic(image_path: str, args):
    """Load and preprocess an image for dynamic-resolution VLM inference.

    Args:
        image_path: Path to the image file.
        args: Megatron args (must have patch_dim, dynamic_resolution_*).

    Returns:
        imgs: Tensor [1, total_patches, patch_features] on CUDA.
        imgs_sizes: Tensor [num_images, 2] with [H, W] in pixels on CUDA.
    """
    from PIL import Image
    from torchvision import transforms as T

    img = Image.open(image_path).convert("RGB")

    # Resize to fit within patch budget.
    patch_dim = args.patch_dim
    pixel_shuffle = getattr(args, 'pixel_shuffle', False)
    min_patches = getattr(args, 'dynamic_resolution_min_patches', 1)
    max_patches = getattr(args, 'dynamic_resolution_max_patches', 128)

    img = dynamic_res_preprocess(
        img,
        min_patches=min_patches,
        max_patches=max_patches,
        res_step=patch_dim,
        pixel_shuffle=pixel_shuffle,
    )

    # Build transform (ToTensor + Normalize only; resize already done above).
    vision_type = getattr(args, 'vision_model_type', 'radio')
    pixel_mean = getattr(args, 'pixel_mean', None)
    pixel_std = getattr(args, 'pixel_std', None)
    if pixel_mean is None or pixel_std is None:
        pixel_mean, pixel_std = _PIXEL_STATS.get(vision_type, (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=pixel_mean, std=pixel_std),
    ])

    img_tensor = transform(img)  # [C, H, W]
    C, H, W = img_tensor.shape

    # Patchify: [C, H, W] -> [num_patches, C*patch_dim*patch_dim]
    py, px = H // patch_dim, W // patch_dim
    patches = img_tensor.reshape(C, py, patch_dim, px, patch_dim)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()
    patches = patches.reshape(py * px, C * patch_dim * patch_dim)

    # images: [1, num_patches, features], imgs_sizes: [1, 2] with [H, W] in pixels
    images = patches.unsqueeze(0)
    imgs_sizes = torch.tensor([[H, W]], dtype=torch.int32)

    return images.cuda(), imgs_sizes.cuda()


def load_and_preprocess_image(image_path: str, args):
    """Load and preprocess an image for VLM inference.

    Args:
        image_path: Path to the image file.
        args: Megatron args.

    Returns:
        imgs: Tensor of image tiles [num_tiles, C, H, W].
        num_tiles: Tensor with number of tiles [1].
    """
    from PIL import Image

    try:
        from examples.multimodal.image_processing import get_visual_transform
    except ImportError:
        from image_processing import get_visual_transform

    img = Image.open(image_path).convert("RGB")

    transform = get_visual_transform(
        img,
        args.img_h,
        args.img_w,
        args.use_tiling,
        args.max_num_tiles,
        args.use_thumbnail,
        args.pixel_mean,
        args.pixel_std,
    )

    if isinstance(transform, tuple):
        imgs, num_tiles_val = transform
    else:
        imgs = transform
        num_tiles_val = imgs.shape[0]

    if not isinstance(imgs, torch.Tensor):
        imgs = torch.tensor(imgs)

    num_tiles = torch.tensor([num_tiles_val], dtype=torch.int)
    return imgs.cuda(), num_tiles.cuda()


# ---------------------------------------------------------------------------
# Inference loops
# ---------------------------------------------------------------------------

def run_text_only_inference(
    requests: List[Request],
    engine: DynamicInferenceEngine,
):
    """Run text-only inference using the dynamic engine."""
    base_arrival_time = get_curr_time()
    for request in requests:
        request.time_arrival = request.time_offset + base_arrival_time

    num_requests_total = len(requests)
    num_requests_added = 0
    total_output_tokens = 0

    while True:
        # Add requests.
        while num_requests_added < num_requests_total:
            if requests[num_requests_added].time_arrival > get_curr_time():
                break
            req = requests[num_requests_added]
            # Same signature as DynamicInferenceEngine.add_request()
            engine.add_request(num_requests_added, req.prompt_text, req.sampling_params)
            req.time_start = get_curr_time()
            req.state = "started"
            num_requests_added += 1

        # Step.
        result = engine.step_modern()
        finished_request_records = result["finished_request_records"]

        for record in finished_request_records:
            finished_request = record.merge()
            request = requests[finished_request.request_id]
            request.time_end = get_curr_time()
            request.state = "finished"
            request.output_tokens = finished_request.generated_tokens
            request.output_text = finished_request.generated_text
            total_output_tokens += len(request.output_tokens)

        if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
            break

    return total_output_tokens


def run_multimodal_inference(
    prompt_entries: List[Dict],
    engine: DynamicInferenceEngine,
    sampling_params: SamplingParams,
    args,
    num_img_embeddings_per_tile: int = 0,
):
    """Run multimodal inference with images.

    Args:
        prompt_entries: List of dicts with 'prompt' and optional 'image' keys.
        engine: The dynamic inference engine.
        sampling_params: Sampling parameters.
        args: Megatron args.
        num_img_embeddings_per_tile: Number of image embeddings per tile (static resolution).
    """
    dynamic_res = getattr(args, 'dynamic_resolution', False)

    # Add all requests.
    for req_id, entry in enumerate(prompt_entries):
        prompt_text = entry["prompt"]
        image_path = entry.get("image")

        if image_path and os.path.exists(image_path):
            if dynamic_res:
                imgs, imgs_sizes = load_and_preprocess_image_dynamic(image_path, args)
                engine.add_request(
                    request_id=req_id,
                    prompt=prompt_text,
                    sampling_params=sampling_params,
                    imgs=imgs,
                    imgs_sizes=imgs_sizes,
                )
            else:
                imgs, num_tiles = load_and_preprocess_image(image_path, args)
                engine.add_request(
                    request_id=req_id,
                    prompt=prompt_text,
                    sampling_params=sampling_params,
                    imgs=imgs,
                    num_tiles=num_tiles,
                    num_img_embeddings_per_tile=num_img_embeddings_per_tile,
                )
        else:
            engine.add_request(
                request_id=req_id,
                prompt=prompt_text,
                sampling_params=sampling_params,
            )

    # Run inference loop.
    finished_records = []
    while engine.has_unfinished_requests():
        result = engine.step_modern()
        finished_records.extend(result.get("finished_request_records", []))

    return finished_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main():
    """Run VLM dynamic inference."""

    # Initialize Megatron.
    initialize_megatron(
        extra_args_provider=add_vlm_inference_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )

    args = get_args()

    # Auto-detect model type from checkpoint.
    is_vlm = _detect_vlm_from_checkpoint(args)
    if torch.distributed.get_rank() == 0:
        print(f"Auto-detected model type: {'VLM' if is_vlm else 'GPT'}")

    # Build tokenizer.
    tokenizer = build_tokenizer(args)

    # Sampling params.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )

    # Load model.
    model = get_model(is_vlm)

    # Build inference config and context.
    inference_config = get_inference_config_from_model_and_args(model, args)

    # For VLM, adjust max_sequence_length to account for image tokens.
    num_img_embeddings_per_tile = 0
    dynamic_res = is_vlm and getattr(args, 'dynamic_resolution', False)
    if is_vlm and hasattr(args, 'img_h') and hasattr(args, 'patch_dim'):
        if dynamic_res:
            # Dynamic resolution: max embeddings = max_patches / 4 (with pixel shuffle).
            max_patches = getattr(args, 'dynamic_resolution_max_patches', 128)
            max_img_embeddings = max_patches
            if getattr(args, 'pixel_shuffle', False):
                max_img_embeddings = max_img_embeddings // 4
            inference_config.max_sequence_length = max(
                inference_config.max_sequence_length,
                max_img_embeddings + args.num_tokens_to_generate + 512,
            )
        else:
            from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings

            num_img_embeddings_per_tile = get_num_image_embeddings(
                args.img_h,
                args.img_w,
                args.patch_dim,
                args.vision_model_type,
                args.disable_vision_class_token,
                1,
                args.pixel_shuffle,
                args.use_tile_tags,
                args.max_num_tiles,
                args.tokenizer_prompt_format,
            )
            max_num_tiles = args.max_num_tiles + int(getattr(args, "use_thumbnail", False))
            max_img_tokens = max_num_tiles * num_img_embeddings_per_tile
            inference_config.max_sequence_length = max(
                inference_config.max_sequence_length,
                max_img_tokens + args.num_tokens_to_generate + 512,
            )

    # No CUDA graphs for VLM V1.
    inference_config.num_cuda_graphs = None

    # Build context, wrapper, controller, engine.
    # A single DynamicInferenceEngine handles both text-only and multimodal
    # requests.  For VLM models, VLMInferenceWrapper provides the vision
    # encoder and image-token expansion; for GPT models it falls through to
    # the standard GPT forward path.
    context = DynamicInferenceContext(model.config, inference_config)
    wrapped_model = VLMInferenceWrapper(model, context)
    controller = TextGenerationController(wrapped_model, tokenizer)
    engine = DynamicInferenceEngine(controller, context)

    # Decide between text-only and multimodal mode.
    if args.input_prompts_json:
        # ------ Multimodal mode: JSON file with prompts + image paths ------
        assert is_vlm, (
            "Multimodal mode (--input-prompts-json) requires a VLM checkpoint."
        )

        with open(args.input_prompts_json) as f:
            prompt_entries = json.load(f)

        finished_records = run_multimodal_inference(
            prompt_entries, engine, sampling_params, args,
            num_img_embeddings_per_tile=num_img_embeddings_per_tile,
        )

        # Print results.
        if torch.distributed.get_rank() == 0:
            results = {}
            for record in finished_records:
                request = record.merge()
                rid = request.request_id
                gen_text = request.generated_text or tokenizer.detokenize(
                    request.generated_tokens
                )
                prompt = request.prompt or tokenizer.detokenize(
                    request.prompt_tokens.tolist()
                )
                print(f"\n--- Request {rid} ---")
                print(f"Prompt: {prompt[:200]}...")
                print(f"Generated: {gen_text}")
                results[rid] = {"prompt": prompt, "generated_text": gen_text}

            if args.output_path:
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output_path}")

    elif args.input_image_path:
        # ------ Single image mode ------
        assert is_vlm, (
            "Single image mode (--input-image-path) requires a VLM checkpoint."
        )

        from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN

        prompt_entries = [
            {
                "prompt": f"{IMAGE_TOKEN}\nDescribe this image in detail.",
                "image": args.input_image_path,
            }
        ]
        finished_records = run_multimodal_inference(
            prompt_entries, engine, sampling_params, args,
            num_img_embeddings_per_tile=num_img_embeddings_per_tile,
        )

        if torch.distributed.get_rank() == 0:
            for record in finished_records:
                request = record.merge()
                gen_text = request.generated_text or tokenizer.detokenize(
                    request.generated_tokens
                )
                print(f"\nGenerated: {gen_text}")

    else:
        # ------ Text-only mode: same as gpt_dynamic_inference.py ------
        requests = build_requests(args, tokenizer, sampling_params)

        t = get_curr_time()
        total_output_tokens = run_text_only_inference(requests, engine)
        torch.cuda.synchronize()
        total_time = get_curr_time() - t

        # Validate all finished.
        for request in requests:
            assert request.state == "finished", (
                f"request.state == '{request.state}' != 'finished'."
            )

        # Print results.
        if torch.distributed.get_rank() == 0:
            unique_prompt_map = defaultdict(list)
            for i, req in enumerate(requests):
                unique_prompt_map[req.prompt_text].append(i)

            for uid, (prompt_text, idxs) in enumerate(unique_prompt_map.items()):
                escaped = prompt_text.replace("\n", "\\n")
                print(f"\n{uid+1}/{len(unique_prompt_map)} [{len(idxs)} reqs] {escaped}")
                for idx in idxs:
                    req = requests[idx]
                    out = (req.output_text or "").replace("\n", "\\n")
                    print(f"  >> [{len(req.output_tokens)} tokens] {out}")

            throughput = total_output_tokens / total_time if total_time > 0 else 0
            print(f"\n~~~ throughput: {throughput:.3f} tok/s, total: {total_time:.3f}s ~~~")

            if args.output_path:
                json_results = {}
                for i, req in enumerate(requests):
                    json_results[i] = {
                        "input_prompt": req.prompt_text,
                        "generated_text": req.output_text,
                        "generated_tokens": req.output_tokens,
                    }
                with open(args.output_path, "w") as f:
                    json.dump(json_results, f, indent=1)
                print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()

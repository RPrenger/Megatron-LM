#!/bin/bash

# VLM dynamic inference - multimodal mode with images.
#
# Pass --input-prompts-json to provide prompts with associated images.
# Pass --input-image-path for a single image with a default prompt.
#
# The JSON format for --input-prompts-json is:
#   [
#     {"prompt": "<image>\nDescribe this image.", "image": "/path/to/img1.jpg"},
#     {"prompt": "<image>\nWhat is in this photo?", "image": "/path/to/img2.jpg"},
#     {"prompt": "What is 2+2?"}  // text-only request in the same batch
#   ]
#
# IMPORTANT: Update --load and --tokenizer-model for your VLM checkpoint.

CUDA_DEVICE_MAX_CONNECTIONS=1
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
NCCL_ALGO=Ring
CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- UPDATE THESE PATHS ----
VLM_CHECKPOINT="/path/to/your/vlm-checkpoint"
TOKENIZER_MODEL="/path/to/your/tokenizer"
PROMPTS_JSON="/path/to/prompts.json"      # or use --input-image-path instead
# IMAGE_PATH="/path/to/image.jpg"         # single image mode

python -m examples.inference.vlm.vlm_dynamic_inference \
--use-mcore-models \
--tokenizer-type TikTokenizer \
--tiktoken-pattern v2 \
--tokenizer-model ${TOKENIZER_MODEL} \
--auto-detect-ckpt-format \
--max-tokens-to-oom 3600000 \
--inference-max-seq-length 4096 \
--attention-backend flash \
--use-checkpoint-args \
--micro-batch-size 1 \
--no-load-optim \
--no-use-tokenizer-model-from-checkpoint-args \
--timing-log-level 2 \
--load ${VLM_CHECKPOINT} \
--distributed-backend nccl \
--log-interval 1 \
--transformer-impl transformer_engine \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--ckpt-format torch_dist \
--bf16 \
--temperature 1.0 \
--top_k 1 \
--num-tokens-to-generate 128 \
--inference-max-requests 16 \
--inference-dynamic-batching \
--input-prompts-json ${PROMPTS_JSON}
# To use single image mode instead, replace the last line with:
# --input-image-path ${IMAGE_PATH}

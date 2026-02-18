#!/bin/bash

# VLM dynamic inference - text-only mode with a GPT checkpoint.
#
# Runs the SAME model and prompts as inference.sh, but through the unified
# dynamic engine (DynamicInferenceEngine + VLMInferenceWrapper).
#
# The checkpoint type (GPT vs VLM) is auto-detected. When a GPT checkpoint is
# loaded, the engine falls through to the standard GPT forward path, so the
# output should be identical to inference.sh.

CUDA_DEVICE_MAX_CONNECTIONS=1
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
NCCL_ALGO=Ring
CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m examples.inference.vlm.vlm_dynamic_inference \
--tiktoken-pattern v2 \
--use-mcore-models \
--tokenizer-type TikTokenizer \
--tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data-quality/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json \
--auto-detect-ckpt-format \
--max-tokens-to-oom 3600000 \
--inference-max-seq-length 4096 \
--attention-backend flash \
--use-checkpoint-args \
--micro-batch-size 1 \
--no-load-optim \
--no-use-tokenizer-model-from-checkpoint-args \
--timing-log-level 2 \
--load /lustre/fsw/portfolios/llmservice/users/ksanthanam/llama3.1-8b-mcore \
--distributed-backend nccl \
--log-interval 1 \
--transformer-impl transformer_engine \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--ckpt-format torch_dist \
--bf16 \
--temperature 1.0 \
--top_k 1 \
--num-tokens-to-generate 30 \
--inference-max-requests 16 \
--prompts "lskjdf" \
--inference-dynamic-batching

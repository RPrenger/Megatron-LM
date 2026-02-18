#!/bin/bash

#HF_HOME=/lustre/fsw/portfolios/llmservice/users/ksanthanam/hf_home
CUDA_DEVICE_MAX_CONNECTIONS=1
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
NCCL_ALGO=Ring
CUBLAS_WORKSPACE_CONFIG=:4096:8

#--tokenizer-model meta-llama/llama-3.1-8b \
#--tokenizer-model /lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data-quality/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json \

python -m examples.inference.gpt.gpt_dynamic_inference \
--tiktoken-pattern v2 \
--use-mcore-models \
--tokenizer-type TikTokenizer \
--enable-cuda-graph \
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
--log-memory-to-tensorboard \
--log-num-zeros-in-grad \
--log-validation-ppl-to-tensorboard \
--log-timers-to-tensorboard \
--temperature 1.0 \
--top_k 1 \
--return-log-probs \
--num-tokens-to-generate 30 \
--dist-ckpt-strictness log_unexpected \
--inference-max-requests 16 \
--prompts "lskjdf" \
--inference-dynamic-batching-num-cuda-graphs 16 \
--inference-dynamic-batching

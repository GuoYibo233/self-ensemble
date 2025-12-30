#!/bin/bash
# filepath: run_parallel_ensemble_all_models_grouped.sh

MODELS=(
    "llama3.1_8b"
    "llama3.1_8b_it"
    "llama3.2_1b"
    "llama3.2_1b_it"
    "llama3.2_3b"
    "llama3.2_3b_it"
    "qwen2.5_14b"
    "qwen2.5_14b_it"
    "qwen2.5_3b"
    "qwen2.5_3b_it"
    "qwen2.5_7b"
    "qwen2.5_7b_it"
    "qwen3_0.6b"
    "qwen3_1.7b"
    "qwen3_30b"
)

DATASET="myriadlama"
SCRIPT="/home/y-guo/self-ensemble/GYB_self-ensemble/parallel_ensemble.py"
GPUS=(0 1 4 5 6 7 8 9)
NUM_GPUS=${#GPUS[@]}

OUTROOT="/home/y-guo/self-ensemble/myriadlama"

# 1. 生成所有任务
TASKS=()
for MODEL in "${MODELS[@]}"; do
  for SHOT in {0..8}; do
    OUTDIR="${OUTROOT}/${MODEL}"
    OUTFILE="${OUTDIR}/myriadlama.logits.avg.${SHOT}samples.10paras.feather"
    if [ -f "$OUTFILE" ]; then
      echo "✅ $OUTFILE exists, skipping."
      continue
    fi
    TASKS+=("$MODEL $SHOT")
  done
done

# 2. 按GPU分组
for ((g=0; g<$NUM_GPUS; g++)); do
  {
    for ((i=g; i<${#TASKS[@]}; i+=NUM_GPUS)); do
      TASK=(${TASKS[$i]})
      MODEL=${TASK[0]}
      SHOT=${TASK[1]}
      OUTDIR="${OUTROOT}/${MODEL}"
      OUTFILE="${OUTDIR}/myriadlama.logits.avg.${SHOT}samples.10paras.feather"
      mkdir -p "$OUTDIR"
      echo "🚀 [GPU ${GPUS[$g]}] Running $MODEL shot=$SHOT"
      CUDA_VISIBLE_DEVICES=${GPUS[$g]} \
      python $SCRIPT \
        --model $MODEL \
        --dataset $DATASET \
        --num_paraphrases 10 \
        --num_samples $SHOT \
        --num_fewshots 1 \
        --logits_ensemble_method avg \
        --rewrite
    done
  } &
done

wait
echo "全部任务已分组并完成。"
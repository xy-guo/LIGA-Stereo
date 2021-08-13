#!/usr/bin/env bash
NGPUS=$1
EXP_NAME=$2
CFG=$3
PY_ARGS=${@:4}

# avoid too many threads
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -x
python -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/train.py \
    --launcher pytorch \
    --fix_random_seed \
    --sync_bn \
    --save_to_file \
    --cfg_file $CFG \
    --exp_name $EXP_NAME \
    ${PY_ARGS}


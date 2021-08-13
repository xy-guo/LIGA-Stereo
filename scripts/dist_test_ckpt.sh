#!/usr/bin/env bash
NGPUS=$1
CFG=$2
CKPT=$3
PY_ARGS=${@:4}

# avoid too many threads
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -x
python -m torch.distributed.launch --nproc_per_node=${NGPUS} tools/test.py \
    --launcher pytorch \
    --save_to_file \
    --cfg_file $CFG \
    --ckpt $CKPT \
    ${PY_ARGS}


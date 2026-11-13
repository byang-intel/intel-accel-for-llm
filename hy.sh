#!/bin/bash -e

#export IAXL_OMP_THREAD_NUM=1
#export IAXL_QAT_DEVICES=0

export IAXL_OMP_THREAD_NUM=4
export IAXL_QAT_DEVICES=0,1,2,3

export TP_SIZE=1
export KVSHRINK_QAT_DEVICES=0,1,2,3

export IAXL_QAT_ZIP_ENABLE=1

export IAXL_CPU_ZIP_ENABLE=0
export IAXL_IAA_ZIP_ENABLE=0
export IAXL_DSA_GD_ENABLE=0

export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0
export IAXL_DDR_POOL_SIZE_GB=1

source setvars.sh
export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

#numactl --cpunodebind=0 --membind=0 python3 benchmark/kvstore/kvstore_hunyuan_test.py --dump-dir hunyuan/int4/merged/ --mla-shuffle planes_cm
#numactl --cpunodebind=0 --membind=0 ./tests/hunyuan-kv-zip-int4.py --mla-shuffle planes_cm --dump-dir hunyuan/int4/merged/
numactl --cpunodebind=0 --membind=0 ./tests/hunyuan-kv-zip-fp8.py --mla-shuffle planes_cm --dump-dir hunyuan/fp8/merged/

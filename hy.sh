#!/bin/bash -e


export TP_SIZE=1
export IAXL_QAT_ZIP_ENABLE=1
export IAXL_IAA_ZIP_ENABLE=0
export IAXL_DSA_GD_ENABLE=0

export IAXL_CPU_ZIP_ENABLE=0
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0
export IAXL_DDR_POOL_SIZE_GB=32

source setvars.sh
export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

#python3 benchmark/kvstore/kvstore_hunyuan_test.py --dump-dir hunyuan/int4/merged/ --mla-shuffle planes_cm
numactl --cpunodebind=0 --membind=0 ./tests/hunyuan-kv-zip-int4.py --mla-shuffle planes_cm --dump-dir hunyuan/int4/merged/

# intel-accel-for-llm (`iaxl`)

`iaxl` 使用 Intel QAT 压缩和可选的 Intel DSA + GDRCopy 加速，为 LLM 推理提供 GPU KV cache 卸载、缓存与持久化能力。

## Build 和 Install

默认构建 CUDA backend。先启动开发容器：

```bash
./start.sh
```

保持容器运行，在另一个终端执行 build 和 editable install：

```bash
./build_in_cnt.sh
```

该脚本在 `iaxl.vllm` 容器内执行：

```bash
pip install -e . --verbose --no-build-isolation
```

## KVStore Benchmark

完成安装后，在宿主机执行默认 benchmark：

```bash
docker exec -it iaxl.vllm benchmark/kvstore/kvstore_benchmark.sh
```

默认配置使用 CUDA，并生成 32 层 BF16 模拟 KV cache 数据。测试会输出 PUT/GET 吞吐、压缩率、cache 统计，并以 `Verification: passed` 表示数据校验成功。需要使用文件数据时，传入 `--data-file PATH`。

启用 Intel DSA + GDRCopy：

```bash
docker exec -it iaxl.vllm benchmark/kvstore/kvstore_benchmark.sh --dsa
```

使用 Nsight Systems 采集 CUDA/NVTX profile：

```bash
docker exec -it iaxl.vllm benchmark/kvstore/kvstore_benchmark.sh --nsys
```

profile 输出到 `_data/nsys_report*`。`--dsa` 和 `--nsys` 可以同时使用。

自定义 shape、层数、dtype 或测试数据时，直接运行 Python benchmark：

```bash
docker exec -it iaxl.vllm bash -c '\
  export LD_PRELOAD=/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}; \
  numactl --cpunodebind=0 --membind=0 \
  python3 benchmark/kvstore/kvstore_benchmark.py \
    --kv-cache-shape 2 1024 16 4 128 \
    --num-layers 32 \
    --dtype bf16'
```

查看全部参数：

```bash
docker exec -it iaxl.vllm python3 benchmark/kvstore/kvstore_benchmark.py --help
```
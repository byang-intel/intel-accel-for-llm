# intel-accel-for-llm (`iaxl`)

`iaxl` 使用 Intel QAT 压缩和可选的 Intel DSA + GDRCopy 加速，为 LLM 推理提供 GPU KV cache 卸载、缓存与持久化能力。

## Build 和 Install

先启动容器：

```bash
./start.sh
```

在容器内执行：

```bash
pip install -e . --verbose --no-build-isolation
```

## KVStore Benchmark

完成安装后，在容器内执行：

```bash
./benchmark/kvstore/kvstore_benchmark.sh
```

默认配置使用 CUDA，并生成 32 层 BF16 模拟 KV cache 数据。测试会输出 PUT/GET 吞吐、压缩率、cache 统计，并以 `Verification: passed` 表示数据校验成功。需要使用文件数据时，传入 `--data-file PATH`。

启用 Intel DSA 加速KV cache传输：

```bash
./benchmark/kvstore/kvstore_benchmark.sh --dsa
```

使用 Nsight Systems 采集 CUDA/NVTX profile：

```bash
./benchmark/kvstore/kvstore_benchmark.sh --nsys
```

profile 输出到 `_data/nsys_report*`。`--dsa` 和 `--nsys` 可以同时使用。

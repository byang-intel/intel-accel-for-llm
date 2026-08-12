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

## KVShrink vLLM Example

KVShrink 是基于 IAXL `KVStore` 的 vLLM V1 KV connector。启动容器前，在宿主机设置模型和 Tensor Parallel 数量：

```bash
MODEL=/media/model-space/Qwen/Qwen3-32B TP_SIZE=4 ./start.sh
```

`setvars.sh` 会根据前 `TP_SIZE` 张 GPU 的 NUMA 拓扑自动生成每个 rank 的 CPU affinity。启用 `IAXL_KV_COMPRESSION` 时会同时生成 QAT device 配置，启用 `IAXL_DSA_GD_ENABLE` 时会生成 DSA work queue 配置。也可以在执行 `start.sh` 前通过以下环境变量覆盖自动配置：

- `VLLM_CPU_OMP_THREADS_BIND`：使用 `|` 分隔每个 rank 的 CPU 列表。
- `KVSHRINK_QAT_DEVICES`：使用 `|` 分隔每个 rank 的 QAT device index 列表。
- `KVSHRINK_DSA_DEVICES`：使用 `|` 分隔每个 rank 的 DSA work queue。
- `IAXL_CACHE_DIR`：持久化 KV cache 的根目录，默认是 `_data/kvcache`。
- `IAXL_DDR_POOL_SIZE_GB`：每个 worker 使用的 DDR cache pool 大小。

需要启用 Intel DSA + GDRCopy 传输时，在启动容器前增加：

```bash
IAXL_DSA_GD_ENABLE=1 MODEL=/media/model-space/Qwen/Qwen3-32B TP_SIZE=4 ./start.sh
```

完成安装后，在容器内启动服务：

```bash
./examples/kvshrink-vllm-serve.sh
```

该脚本会在 `localhost:8000` 启动 vLLM，加载 `KVShrinkConnector`，并将日志写入 `log.kvshrink-vllm`。启动日志中的 `MODEL`、`TP_SIZE` 和每个 rank 的 CPU/QAT/DSA 配置可用于检查实际生效的拓扑。

在宿主机的第二个终端进入同一个容器：

```bash
docker exec -it -w "$PWD" iaxl.vllm bash
```

先确认模型已经注册并发送一个 Chat Completions 请求：

```bash
curl http://localhost:8000/v1/models
./tests/vllm-test.sh
```

请求中的 `model` 必须与服务启动时的 `MODEL` 完全一致。使用本地模型路径启动时，测试和 benchmark 也应使用相同路径，例如：

```bash
MODEL=/media/model-space/Qwen/Qwen3-32B ./tests/vllm-test.sh
```

## KVShrink vLLM Benchmark

保持 KVShrink vLLM 服务运行，在第二个容器终端执行 online serving benchmark：

```bash
MODEL=/media/model-space/Qwen/Qwen3-32B ./tests/vllm-benchmark.sh
```

benchmark 脚本使用 `vllm bench serve`，默认负载如下：

- 5 个 warmup 请求和 10 个正式请求。
- 每个请求包含 8000 个输入 token 和 128 个输出 token。
- 输入由 6400 个固定公共前缀 token 和 1600 个随机 token 组成，用于构造约 80% 的可复用前缀。
- `request-rate=inf`，所有请求立即提交，最大并发数为 4。
- 输出 TTFT 和 TPOT 的 P50/P95，以及 request 和 token throughput。

其中 TTFT（Time To First Token）包含请求调度、外部 KV cache 查询与加载等首 token 路径开销；TPOT（Time Per Output Token）用于观察 decode 阶段性能。对比 KVShrink 配置时，应保持模型、输入输出长度、并发数和 cache 初始状态一致。

多轮独立 benchmark 会持续占用 DDR cache。开始新一轮冷 cache 测试前，可以清空当前内存缓存：

```bash
curl -X POST http://localhost:18700/v1/cache/evict \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
```

需要保留 cache 数据时，先持久化到磁盘再清理 DDR：

```bash
curl -X POST http://localhost:18700/v1/cache/persist \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
curl -X POST http://localhost:18700/v1/cache/evict \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
```

输入长度、输出长度、请求数量、并发数和公共前缀比例定义在 `tests/vllm-benchmark.sh` 开头，可按测试场景调整。完整的 vLLM benchmark 参数可通过以下命令查看：

```bash
vllm bench serve --help=all
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

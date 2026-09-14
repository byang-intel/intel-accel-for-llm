[English](README.md)

# 性能测试

请在配置好的开发环境中，从仓库根目录运行以下命令。

## KVStore

测试 KVStore 的 PUT/GET 带宽、压缩率以及压缩/解压吞吐量。

```bash
bash benchmark/kvstore/kvstore_benchmark.sh
```

通过 `--dsa` 或 `--nsys` 启用 DSA 或 Nsight Systems 分析：

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --dsa
python3 benchmark/kvstore/kvstore_benchmark.py --shape 2 1024 16 4 128 --dtype bf16 --num-layers 32
```

默认使用 `$MODEL` 真实模型预填充生成 KV cache 数据，结果以 `<model>_<dtype>.pt` 保存在 `--kv-data-dir` 中，shape 匹配时直接复用。`--dtype` 支持 `bf16`、`fp8_e4m3`（per-tensor k/v scale）和 `int4`（对称 group-wise，每字节两个 4bit 值），量化方式与 vLLM 保持一致。使用 `--data-source mock` 可改用合成数据：

```bash
python3 benchmark/kvstore/kvstore_benchmark.py --dtype int4 \
    --prompt-file /path/to/text.txt --model-seq-len 16384
```

## Tensor Transfer

对比 CUDA、`cudaMemcpy3DBatchAsync`、Triton 和 IAXL 的碎片化 H2D/D2H 传输性能，并生成结果图。

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh
```

用 `--direction h2d` 或 `--direction d2h` 只跑单个方向，用 `--methods` 只跑部分方法（`cuda`、`batch`、`triton`、`iaxl`）：

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --direction h2d --methods iaxl cuda
```

所有生成的文件都放在 `/_data/tensor_xfer_benchmark`，可用 `--output-dir` 修改。

加上 `--flamegraph` 可使用 `perf record` 采样。perf 通过 control FIFO 控制，因此只采集计时迭代（不含 warmup），并覆盖进程的所有线程，包括 IAXL DSA 的原生工作线程。运行后会生成火焰图 SVG、供 `flamegraph.pl` 或 speedscope 使用的 `.folded` 文件，并保留原始 `perf.data` 以供 `perf report` 分析。

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --flamegraph tensor_xfer_flame.svg
```

要求容器内已安装 `perf`，且宿主机 `kernel.perf_event_paranoid <= 2`。二进制文件未保留帧指针时可用 `--flamegraph-call-graph dwarf`，`--flamegraph-freq` 用于调整采样频率。采样会干扰计时，因此仅用于分析，不要用于上报性能数据。

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --flamegraph tensor_xfer_flame.svg
```
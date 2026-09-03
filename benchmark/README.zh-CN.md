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
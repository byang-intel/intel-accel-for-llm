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

## Tensor Transfer

对比 CUDA、`cudaMemcpy3DBatchAsync`、Triton 和 IAXL 的碎片化 H2D/D2H 传输性能，并生成结果图。

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh
```
[中文](README.zh-CN.md)

# Benchmarks

Run the following commands from the repository root in the configured development environment.

## KVStore

Measures KVStore PUT/GET bandwidth, compression ratio, and compression/decompression throughput.

```bash
bash benchmark/kvstore/kvstore_benchmark.sh
```

Enable DSA or Nsight Systems profiling with `--dsa` or `--nsys`:

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --dsa
python3 benchmark/kvstore/kvstore_benchmark.py --shape 2 1024 16 4 128 --dtype bf16 --num-layers 32
```

## Tensor Transfer

Compares fragmented H2D and D2H transfer performance using CUDA, `cudaMemcpy3DBatchAsync`, Triton, and IAXL, and generates a result plot.

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh
```
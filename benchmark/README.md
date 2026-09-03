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

By default the KV cache is filled by a real transformer prefill of `$MODEL`. The generated data is stored as `<model>_<dtype>.pt` under `--kv-data-dir` and reused whenever it matches the requested shape. `--dtype` selects `bf16`, `fp8_e4m3` (per-tensor k/v scale), or `int4` (symmetric group-wise, two 4-bit values per byte), matching vLLM's quantization. Use `--data-source mock` for synthetic data:

```bash
python3 benchmark/kvstore/kvstore_benchmark.py --dtype int4 \
    --prompt-file /path/to/text.txt --model-seq-len 16384
```

## Tensor Transfer

Compares fragmented H2D and D2H transfer performance using CUDA, `cudaMemcpy3DBatchAsync`, Triton, and IAXL, and generates a result plot.

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh
```
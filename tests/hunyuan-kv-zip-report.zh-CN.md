# Hunyuan MLA KV Cache 压缩测试报告与集成建议

> 测试对象：IAXL `kv_zip`（QAT DEFLATE）在腾讯混元（Hunyuan）模型 INT4 / FP8 MLA KV cache 上的压缩率与吞吐。
> 测试脚本：[tests/hunyuan-kv-zip-int4.py](hunyuan-kv-zip-int4.py)、[tests/hunyuan-kv-zip-fp8.py](hunyuan-kv-zip-fp8.py)

---

## 1. 测试条件与环境

### 1.1 硬件

| 项目 | 配置 |
| --- | --- |
| CPU | Intel® Xeon® 6767P（Granite Rapids），2 路，每路 64 核 / 128 线程，4 个 NUMA 节点 |
| 内存 | 503 GiB DDR5 |
| 压缩加速器 | Intel® QAT 4xxx（设备 ID 8086:4944），每路 4 个 QAT 设备，整机 8 个 |
| 本次使用 | **1 路 CPU（NUMA node 0）上的 4 个 QAT 设备**，`numactl --cpunodebind=0 --membind=0` 绑定 |

### 1.2 软件

| 项目 | 版本 / 设置 |
| --- | --- |
| 宿主机内核 | Linux 6.8.0-71-generic |
| 容器 | `iaxl.vllm`，Ubuntu 24.04.3 LTS，PyTorch 2.13.0 |
| 压缩后端 | `IAXL_QAT_ZIP_ENABLE=1`，`IAXL_CPU_ZIP_ENABLE=0`，`IAXL_IAA_ZIP_ENABLE=0` |
| QAT 配置 | `IAXL_QAT_DEVICES=0,1,2,3`，每设备 4 个实例（共 16 个驱动线程），队列深度 4 |
| DEFLATE 级别 | `IAXL_ZIP_LEVEL=1` |

本次 dump 以 `torch.uint8` 原始字节保存，因此 C++ 侧 shuffle 未介入，所有重排均由 Python 侧临时实现（详见第 4 节）。

### 1.3 数据集

数据来自混元模型真实推理过程中 dump 的 KV cache（84,412 个 prompt token，未截断），每层一个 `layer_NNN.pt`，共 78 层。格式文档：
KV_CACHE_DUMP_FORMAT.md

| 数据集 | 每 token 行布局（字节） | 行长 | 78 层 MLA 总量 |
| --- | --- | --- | --- |
| INT4 | `[0,256)` 512 个 int4（偏移 +8）· `[256,320)` 32 个 bf16 scale · `[320,448)` 64 个 bf16 RoPE | 448 B | 2.75 GiB |
| FP8 | `[0,512)` 512 个 fp8 e4m3 · `[512,528)` 4 个 fp32 scale · `[528,656)` 64 个 bf16 RoPE | 656 B | 4.02 GiB |


### 1.4 测试方法

- 每层的 MLA cache 按 **64 token/block** 切分（`block_size=64`，INT4 块 28 KiB，FP8 块 41 KiB），每个 block 作为一个独立的压缩单元，与 KVStore 的实际压缩粒度一致。
- 流程：加载 → block 内字节重排（可选）→ `torch_ext.compress_tensors` → `torch_ext.decompress_tensors` → 逆重排 → 与原始字节 `torch.equal` 逐字节比对。
- 计时只覆盖 `compress_tensors` / `decompress_tensors` 两个调用；正式计时前先做一次 warmup；每层每个模式重复 5 次（`--iters 5`）取累计耗时，避免单次耗时过短带来的抖动。
- 吞吐以**未压缩字节数 / 耗时**计算，单位 GB/s（1 GB = 10⁹ B）。压缩率以 tensor 原始字节数为基准。
- 三种数据布局模式：
  - `none`：dump 原始布局，直接压缩；
  - `planes`：block 内按字节平面重排（int4/fp8 平面、scale 高/低字节平面、RoPE 高/低字节平面）；
  - `planes_cm`：在 `planes` 基础上，各平面内部改为 channel-major（同一通道跨 token 连续）。
  
  重排是**纯字节置换，无损**，不改变任何字节值、不增减字节数；原理见测试脚本的模块注释。

---

## 2. 性能（吞吐）

1 路 CPU + 4 个 QAT 设备（16 个 QAT 实例），78 层全量，无损：

### 2.1 INT4（2.75 GiB）

| 布局模式 | 压缩吞吐 | 解压吞吐 |
| --- | ---: | ---: |
| none | 11.36 GB/s | 19.40 GB/s |
| planes | 11.97 GB/s | 20.07 GB/s |
| planes_cm | 11.41 GB/s | 20.02 GB/s |

### 2.2 FP8（4.02 GiB）

| 布局模式 | 压缩吞吐 | 解压吞吐 |
| --- | ---: | ---: |
| none | 12.92 GB/s | 20.27 GB/s |
| planes | 13.62 GB/s | 21.26 GB/s |
| planes_cm | 13.32 GB/s | 21.43 GB/s |

### 2.3 观察

- 单路 4 个 QAT 设备可提供 **约 11–13 GB/s 压缩、约 20–21 GB/s 解压** 的稳定吞吐，解压约为压缩的 1.6–1.8 倍。
- 吞吐与数据布局基本无关（波动 < 6%）：字节重排不会给 QAT 增加负担，反而因为高字节平面更易压缩而略有提升。
- 作为对照，仅用 1 个 QAT 设备（4 实例）时同一测试约为 4 GB/s 压缩、5.5 GB/s 解压，吞吐随 QAT 设备数近似线性扩展，整机 8 个设备可预期翻倍。
- 该吞吐远高于典型 KV cache 落盘 / 跨节点传输带宽，压缩不会成为 KV 缓存链路的瓶颈。

---

## 3. 压缩率

### 3.1 结果

| 数据集 | 布局模式 | 压缩率 | 节省空间 |
| --- | --- | ---: | ---: |
| INT4 | none | 1.106× | 9.6% |
| INT4 | planes | 1.187× | 15.8% |
| INT4 | **planes_cm** | **1.204×** | **16.9%** |
| FP8 | none | 1.166× | 14.2% |
| FP8 | planes | 1.212× | 17.5% |
| FP8 | **planes_cm** | **1.222×** | **18.2%** |

全部 78 层 × 3 种模式的压缩 / 解压回环均逐字节校验通过。

### 3.2 分析

**为什么原始布局压缩率偏低。** 对 INT4 layer_001 的熵分析显示：

- int4 nibble 在 1..15 上接近均匀分布（约 3.9 bit / 4 bit），占行长的 57%，本质上不可压缩；
- bf16 的低字节（1 位指数 + 7 位尾数）接近随机噪声；
- 真正可压的是 bf16 高字节（符号 + 指数高 7 位）：同一层内数值量级集中，指数只取少数几个值；
- 原始布局把这三类统计特性截然不同的字节按 token 交错在一起，DEFLATE 的 Huffman 只能用一张"平均"的码表，LZ77 也很难找到长匹配，整行字节熵约 6.96 bit，理论上限仅约 1.15×。

**字节平面重排的作用。** 把同类字节集中到连续区域后，高字节平面的局部熵远低于 8 bit，可以用很短的 Huffman 码；低字节和 nibble 平面虽仍不可压，但不再"稀释"其它平面。channel-major 进一步把同一通道跨 token 的字节排在一起——RoPE 是位置编码，同一维度随位置缓慢变化，相邻 token 的高字节经常相同，形成 LZ77 可匹配的重复串。实测：INT4 由 1.106× 提升到 1.204×（节省空间 +76%），FP8 由 1.166× 提升到 1.222×。

**FP8 与 INT4 的差异。** FP8 原始布局压缩率略高（fp8 字节自带 4 位指数，分布比 int4 nibble 集中），但重排带来的增益较小（+4.8% vs +8.9%），原因是 78% 的字节是单字节 fp8，拆不出"高/低字节"平面；fp32 scale 仅占 2.4%，重排主要是避免其随机尾数污染其它平面。

---

## 4. 集成建议

### 4.1 现状

- IAXL `kv_zip` 已内置 data shuffle优化，但其实现针对 **bf16 K/V cache**，对 uint8 原始字节（混元 INT4 / FP8 dump 的保存格式）不生效。
- 本次测试中，面向混元 MLA 布局的 `planes` / `planes_cm` 重排是在 Python 侧用 `torch` 临时实现的（`shuffle_mla_blocks` / `shuffle_fp8_blocks`），用于在**不改 C 代码**的前提下验证收益。它已经证明：针对混元布局的字节重排可以无损地把节省空间从 ~10% 提升到 ~17–18%，而且不影响 QAT 吞吐。

### 4.2 建议：在 C++ 中集成混元布局的 data shuffle

Python 侧重排只适合验证，正式集成应下沉到 `iaxl/csrc/kv_zip`，原因有三：避免 Python 层额外一次完整内存拷贝和 GIL 开销；可以与现有分段压缩流水线、QAT 输入缓冲区准备阶段合并（在 `shuffle_copy` 填充 QAT 源缓冲时顺带完成重排，几乎零成本）；对调用方透明，压缩块头部自描述、解压侧自动逆变换。

### 4.3 作为独立 Python 模块接入腾讯 KV 缓存方案

IAXL 的压缩能力已经以 C 扩展形式暴露给 Python（`iaxl.torch_ext`），可以**不依赖 vLLM / KVStore**，直接嵌入腾讯自有的 KV cache 缓存方案：

```python
from iaxl import torch_ext

# blocks: list[torch.Tensor]，CPU 连续、同 dtype、同大小，每个 tensor 是一个 KV block
zipped = torch_ext.compress_tensors(blocks)          # -> list[torch.uint8]，每个 block 一段自描述的压缩数据
torch_ext.decompress_tensors(zipped, out_blocks)     # 原地解压到预分配的 out_blocks
```

接口特点与注意事项：

- 输入 / 输出均为 `torch.Tensor`，与 PyTorch 原生 KV cache 零拷贝对接；也可以用 `torch.frombuffer` 包装任意 `bytes` / `numpy` 缓冲区。
- 压缩数据块自带头部（原始长度、分段表、shuffle 标志），可以直接作为缓存条目落盘或经网络传输，解压端不需要额外元数据。
- 后端通过环境变量选择：QAT（`IAXL_QAT_ZIP_ENABLE`）、IAA（`IAXL_IAA_ZIP_ENABLE`）、CPU 软件 DEFLATE（`IAXL_CPU_ZIP_ENABLE`），三者输出格式兼容，可按部署机型自由切换；无 QAT 的机器可用 CPU 后端解压 QAT 压缩的数据。
- `compress_tensors` 会把输入 tensor 用作分段压缩的 scratch 空间，调用方若之后仍需要原始数据，应传入副本。
- 批量调用（一次传入多个 block）才能充分利用 16 个 QAT 实例的并发；建议每次调用携带数十到上千个 block。
- 待 4.2 的 C++ shuffle 集成完成后，调用方只需在初始化时声明布局（如 `IAXL_KV_LAYOUT=hunyuan_int4`），无需在 Python 侧做任何数据预处理。

### 4.4 后续工作

| 优先级 | 事项 |
| --- | --- |
| 高 | 在 `kv_zip` C++ 中实现混元 INT4 / FP8 布局的 block 内字节平面 + channel-major shuffle，并用现有 Python 实现做回归比对 |
| 高 | 在 `BlockHeader` 中加入布局 ID，保证压缩数据自描述 |

---

## 附录：复现命令

```bash
# 宿主机，仓库根目录（容器 iaxl.vllm 已由 ./start.sh 启动）
docker exec -w "$PWD" iaxl.vllm bash hy.sh
```

`hy.sh` 关键设置：

```bash
export IAXL_OMP_THREAD_NUM=4
export IAXL_QAT_DEVICES=0,1,2,3
export IAXL_QAT_ZIP_ENABLE=1 IAXL_CPU_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0
source setvars.sh
export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

numactl --cpunodebind=0 --membind=0 ./tests/hunyuan-kv-zip-int4.py --dump-dir hunyuan/int4/merged/
numactl --cpunodebind=0 --membind=0 ./tests/hunyuan-kv-zip-fp8.py  --dump-dir hunyuan/fp8/merged/
```

不加 `--mla-shuffle` 时默认依次测试 `none planes planes_cm` 三种模式；`--layers`、`--max-blocks`、`--iters` 可用于缩小规模或调整重复次数。

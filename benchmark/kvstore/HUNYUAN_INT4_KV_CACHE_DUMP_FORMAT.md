# INT4 KV Cache dump 简要说明

## 数据说明

- 输入：`84k-input-request.txt`，完整 **84,412 tokens**，未截断。
- **78 个逐层文件**：`layer_000.pt` 至 `layer_077.pt`，附 `manifest.json`。
- 包含 **78 份 INT4 MLA KV + 21 份 FP8 索引器缓存**，共 99 个 tensor。
- 原始 tensor 共 **3,183,682,992 bytes，约 3.18 GB**；`.pt` 文件合计 3,184,246,589 bytes。
- 已合并全部 CP4 owner 分片，直接使用 `merged/`，无需再次拼接。

## 文件格式

每个 `.pt` 文件由 `torch.save` 保存一个字典：

- `layer`：层号。
- `valid_tokens`：有效 token 数，本批为 `84412`。
- `kv_cache_dtype`：`int4_mla_sm100`。
- `caches`：缓存名称 → 条目，包含 `tensor`、`layout`、`nbytes`、`sha256` 等。

`tensor` 均为 CPU 上连续的 `torch.uint8`，用于保留量化缓存的**原始字节**，不是 INT8 量化，也未反量化。数值使用小端字节序。

### MLA KV

每层形状为 `[84412, 448]`，每行对应一个 token：

| 字节区间 | 内容 |
| --- | --- |
| `[0, 256)` | 512 个 INT4 值，两个 4-bit 值打包为一个 byte |
| `[256, 320)` | 32 个 BF16 scale，每个对应连续 16 个 INT4 值 |
| `[320, 448)` | 64 个 BF16 RoPE 值，未做 INT4 量化 |

INT4 量化值范围为 `[-8, 7]`，以 `nibble = q + 8` 保存，**不是补码编码，也不是 NVFP4 浮点编码**。

MLA 不是传统的独立 K、V 两个 tensor，不要额外乘以 2。做压缩测试无需解包，直接使用原始 bytes。

### 索引器缓存

索引器仍是 **FP8 key + FP32 scale**，不是 INT4。仅部分层存在，形状为 `[11142384]`：

- 每页 64 个 token，先放 8192 bytes FP8 key，再放 256 bytes FP32 scales。
- 本批尾页 60 个 token，先放 7680 bytes key，再放 240 bytes scales。
- 不保存尾页空槽，不能 reshape 为 `[84412, 132]` 后逐行解析 scale。

## 加载示例


```python
from pathlib import Path
import hashlib
import torch

root = Path("./merged")
data = torch.load(root / "layer_000.pt", map_location="cpu", weights_only=True)

for name, entry in data["caches"].items():
    tensor = entry["tensor"]
    raw = tensor.numpy().tobytes()
    assert len(raw) == entry["nbytes"]
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    print(name, tensor.shape, len(raw))
```

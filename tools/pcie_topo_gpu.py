#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_pcie_topo.py - 显示所有 GPU 的 PCIe 拓扑、每一段链路的单向带宽(理论值),
并实测 H2D / D2H 带宽(单卡、同上行组并发、全卡并发)。

依赖:
  拓扑部分: 仅需 Linux sysfs (可选 lspci 以显示设备名)
  测速部分: python3 + PyTorch(CUDA)

用法:
  python3 gpu_pcie_topo.py                  # 拓扑 + 测速
  python3 gpu_pcie_topo.py --no-bench       # 只看拓扑
  python3 gpu_pcie_topo.py --size-mb 512 --iters 30
  python3 gpu_pcie_topo.py --vendor 0x1002  # 其他厂商(默认 NVIDIA 0x10de)

说明:
  * 带宽为"每方向"的理论值(扣除 128b/130b 编码),实际可达约 85%~92%。
  * sysfs 里每个设备的 current_link_speed/width 描述的是"该设备与其上游端口之间"的链路。
  * GPU 空闲时链路会降速省电,脚本默认先做一次短暂拷贝唤醒(--no-wake 关闭)。
"""
import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # 与 nvidia-smi 编号一致

SYSFS = "/sys/bus/pci/devices"
BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
GEN_OF = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


# ----------------------------------------------------------------- sysfs 工具
def rd(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def parse_int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def parse_speed(s):
    m = re.match(r"\s*([\d.]+)\s*GT/s", s or "")
    return float(m.group(1)) if m else 0.0


def parse_cpulist(s):
    out = set()
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def bw_per_dir(gts, width):
    """PCIe 单方向理论带宽 GB/s"""
    if gts <= 0 or width <= 0:
        return 0.0
    if gts <= 5.0:
        eff = 8 / 10          # Gen1/2: 8b/10b
    elif gts <= 32.0:
        eff = 128 / 130       # Gen3/4/5: 128b/130b
    else:
        eff = 242 / 256       # Gen6: FLIT 近似
    return gts * width * eff / 8


@dataclass
class Link:
    speed: float
    width: int
    max_speed: float
    max_width: int

    @property
    def bw(self):
        return bw_per_dir(self.speed, self.width)

    @property
    def degraded(self):
        return (self.speed and self.speed < self.max_speed) or \
               (self.width and self.width < self.max_width)

    @staticmethod
    def _fmt(speed, width):
        if speed <= 0 or width <= 0:
            return "n/a"
        return f"Gen{GEN_OF.get(speed, '?')}x{width}"

    def cur(self):
        return self._fmt(self.speed, self.width)

    def mx(self):
        return self._fmt(self.max_speed, self.max_width)


def link_of(bdf):
    d = f"{SYSFS}/{bdf}"
    return Link(parse_speed(rd(d + "/current_link_speed")),
                parse_int(rd(d + "/current_link_width")),
                parse_speed(rd(d + "/max_link_speed")),
                parse_int(rd(d + "/max_link_width")))


def chain_of(bdf):
    """从 Root Port 到该设备的 BDF 链"""
    real = os.path.realpath(f"{SYSFS}/{bdf}")
    return [p for p in real.split("/") if BDF_RE.match(p)]


def children_of(bdf):
    real = os.path.realpath(f"{SYSFS}/{bdf}")
    try:
        return sorted(x for x in os.listdir(real)
                      if BDF_RE.match(x) and os.path.isdir(f"{real}/{x}"))
    except OSError:
        return []


def is_bridge(bdf):
    return rd(f"{SYSFS}/{bdf}/class").startswith("0x0604")


def lspci_names():
    names = {}
    try:
        out = subprocess.run(["lspci", "-D", "-nn"], capture_output=True,
                             text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return names
    for line in out.splitlines():
        bdf, _, rest = line.partition(" ")
        rest = re.sub(r"\s*\(rev [0-9a-f]+\)", "", rest)
        names[bdf] = rest
    return names


def dev_name(bdf, names):
    if bdf in names:
        s = names[bdf]
    else:
        s = f"{rd(f'{SYSFS}/{bdf}/vendor')}:{rd(f'{SYSFS}/{bdf}/device')}"
    return s if len(s) <= 64 else s[:61] + "..."


# ----------------------------------------------------------------- GPU 发现
@dataclass
class GPU:
    bdf: str
    idx: int                      # 按 PCI 总线排序的编号(与 nvidia-smi 一致)
    chain: list = field(default_factory=list)
    cuda: int = None              # torch 中的设备号
    numa: str = "?"
    cpus: set = field(default_factory=set)


def discover_gpus(vendor):
    gpus = []
    for bdf in sorted(os.listdir(SYSFS)):
        d = f"{SYSFS}/{bdf}"
        if rd(d + "/vendor").lower() != vendor.lower():
            continue
        if not rd(d + "/class").startswith("0x03"):   # 显示/3D 控制器
            continue
        g = GPU(bdf=bdf, idx=len(gpus), chain=chain_of(bdf),
                numa=rd(d + "/numa_node", "?"),
                cpus=parse_cpulist(rd(d + "/local_cpulist")))
        gpus.append(g)
    return gpus


def map_torch(gpus):
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    by_bdf = {g.bdf: g for g in gpus}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        if bdf in by_bdf:
            by_bdf[bdf].cuda = i
    return torch


def wake_gpus(torch, gpus):
    """短暂拷贝,让 PCIe 链路脱离省电降速状态"""
    ctx = []
    for g in gpus:
        if g.cuda is None:
            continue
        with torch.cuda.device(g.cuda):
            h = torch.empty(64 << 20, dtype=torch.uint8, pin_memory=True)
            d = torch.empty(64 << 20, dtype=torch.uint8, device=f"cuda:{g.cuda}")
            for _ in range(20):
                d.copy_(h, non_blocking=True)
                h.copy_(d, non_blocking=True)
            ctx.append((h, d))
    torch.cuda.synchronize()
    return ctx  # 保持引用直到拓扑读取完毕


# ----------------------------------------------------------------- 拓扑打印
def role_of(bdf, parent_role):
    if parent_role is None:
        return "RootPort"
    if is_bridge(bdf):
        return "SW-Down" if parent_role == "SW-Up" else "SW-Up"
    return "EP"


def print_tree(root, gpu_by_bdf, names):
    def rec(bdf, prefix, last, parent_role, top=False):
        role = role_of(bdf, parent_role)
        conn = "" if top else ("└─ " if last else "├─ ")
        tag = f" <<GPU{gpu_by_bdf[bdf].idx}>>" if bdf in gpu_by_bdf else ""
        if role == "RootPort":
            numa = rd(f"{SYSFS}/{bdf}/numa_node", "?")
            info = f"(CPU 侧, NUMA {numa})"
        else:
            lk = link_of(bdf)
            info = f"{lk.cur()}  {lk.bw:6.1f} GB/s/dir"
            if lk.degraded:
                info += f"  ⚠ 降速(最大 {lk.mx()})"
        print(f"{prefix}{conn}[{role}] {bdf}{tag}  {dev_name(bdf, names)}")
        print(f"{prefix}{'' if top else ('   ' if last else '│  ')}"
              f"{'   ' if not top else ''}   链路: {info}")
        kids = children_of(bdf)
        cp = prefix + ("" if top else ("   " if last else "│  "))
        if role == "SW-Up" and kids:
            up = link_of(bdf).bw
            down = sum(link_of(k).bw for k in kids)
            if up > 0:
                print(f"{cp}   ⤷ 上行 {up:.1f} GB/s/向, 下行合计 {down:.1f} GB/s/向"
                      f" → 超额订阅 {down / up:.1f}:1")
        for i, k in enumerate(kids):
            rec(k, cp, i == len(kids) - 1, role)

    rec(root, "", True, None, top=True)


def print_topology(gpus, names):
    print("=" * 78)
    print(" PCIe 拓扑 (带宽为每方向理论值)")
    print("=" * 78)
    by_bdf = {g.bdf: g for g in gpus}
    roots = []
    for g in gpus:
        if g.chain and g.chain[0] not in roots:
            roots.append(g.chain[0])
    for r in roots:
        print_tree(r, by_bdf, names)
        print()

    # 共享上行分析
    print("=" * 78)
    print(" GPU 共享上行分析")
    print("=" * 78)
    groups = {}
    for g in gpus:
        key = g.chain[1] if len(g.chain) >= 3 else g.chain[-1]
        groups.setdefault(key, []).append(g)
    for key, ms in groups.items():
        up = link_of(key)
        down = sum(link_of(m.bdf).bw for m in ms)
        ratio = down / up.bw if up.bw else 0
        who = ",".join(f"GPU{m.idx}" for m in ms)
        print(f" 组[{who}]  共享上行 {key} {up.cur()} = {up.bw:.1f} GB/s/向; "
              f"GPU 下行合计 {down:.1f} GB/s/向 → {ratio:.1f}:1")
    if any(link_of(g.bdf).degraded for g in gpus):
        print("\n ⚠ 存在降速链路: 可能是空闲省电降速,请在有负载时复查(默认已做唤醒)。")
    print()
    return groups


# ----------------------------------------------------------------- 测速
class Ctx:
    def __init__(self, torch, gpu, nbytes):
        self.t = torch
        self.gpu = gpu
        self.i = gpu.cuda
        old = os.sched_getaffinity(0)
        try:                                  # 让 pinned 内存落在 GPU 本地 NUMA
            if gpu.cpus:
                os.sched_setaffinity(0, gpu.cpus)
        except OSError:
            pass
        with torch.cuda.device(self.i):
            self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            self.host.fill_(1)
            self.dev = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{self.i}")
            self.dev.fill_(1)
            self.stream = torch.cuda.Stream(device=self.i)
        os.sched_setaffinity(0, old)

    def enqueue(self, direction, n):
        with self.t.cuda.device(self.i), self.t.cuda.stream(self.stream):
            for _ in range(n):
                if direction == "h2d":
                    self.dev.copy_(self.host, non_blocking=True)
                else:
                    self.host.copy_(self.dev, non_blocking=True)


def measure(torch, ctxs, direction, nbytes, iters):
    """返回 ({gpu.idx: GB/s}, 聚合 GB/s)"""
    for c in ctxs:
        c.enqueue(direction, 3)
    for c in ctxs:
        c.stream.synchronize()
    evs = {}
    t0 = time.perf_counter()
    for c in ctxs:
        with torch.cuda.device(c.i), torch.cuda.stream(c.stream):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            c.enqueue(direction, iters)
            e.record()
            evs[c.gpu.idx] = (s, e)
    for c in ctxs:
        c.stream.synchronize()
    wall = time.perf_counter() - t0
    per = {k: nbytes * iters / (s.elapsed_time(e) / 1e3) / 1e9
           for k, (s, e) in evs.items()}
    agg = nbytes * iters * len(ctxs) / wall / 1e9
    return per, agg


def run_bench(torch, gpus, groups, size_mb, iters):
    nbytes = size_mb << 20
    usable = [g for g in gpus if g.cuda is not None]
    if not usable:
        print("未找到可用的 CUDA 设备(检查 CUDA_VISIBLE_DEVICES)。")
        return
    print("=" * 78)
    print(f" H2D / D2H 带宽实测  (每次 {size_mb} MB × {iters} 次, pinned memory)")
    print("=" * 78)
    ctx = {g.idx: Ctx(torch, g, nbytes) for g in usable}
    res = {g.idx: {} for g in usable}

    # 1) 单卡独占
    print(" [1/3] 单卡独占 ...", flush=True)
    for g in usable:
        for d in ("h2d", "d2h"):
            per, _ = measure(torch, [ctx[g.idx]], d, nbytes, iters)
            res[g.idx][("alone", d)] = per[g.idx]

    # 2) 同上行组并发
    print(" [2/3] 同上行组并发 ...", flush=True)
    grp_notes = []
    for key, ms in groups.items():
        ms = [m for m in ms if m.cuda is not None]
        if len(ms) < 2:
            continue
        for d in ("h2d", "d2h"):
            per, agg = measure(torch, [ctx[m.idx] for m in ms], d, nbytes, iters)
            for m in ms:
                res[m.idx][("group", d)] = per[m.idx]
        who = ",".join(f"GPU{m.idx}" for m in ms)
        for d in ("h2d", "d2h"):
            alone = sum(res[m.idx][("alone", d)] for m in ms) / len(ms)
            grp = sum(res[m.idx][("group", d)] for m in ms) / len(ms)
            grp_notes.append((who, d.upper(), alone, grp))

    # 3) 全卡并发
    all_agg = {}
    if len(usable) > 1:
        print(" [3/3] 全部 GPU 并发 ...", flush=True)
        for d in ("h2d", "d2h"):
            per, agg = measure(torch, [ctx[g.idx] for g in usable], d, nbytes, iters)
            all_agg[d] = agg
            for g in usable:
                res[g.idx][("all", d)] = per[g.idx]

    # 结果表
    def f(g, k, d):
        v = res[g.idx].get((k, d))
        return f"{v:7.1f}" if v is not None else "      -"

    print()
    print(" 单位: GB/s (每卡)         | 单卡独占      | 同上行组并发  | 全卡并发")
    print(" GPU  PCI Addr      Link   |   H2D    D2H  |   H2D    D2H  |   H2D    D2H")
    print(" " + "-" * 74)
    for g in usable:
        lk = link_of(g.bdf)
        print(f" {g.idx:>3}  {g.bdf[5:]:<12} {lk.cur():<8}|"
              f"{f(g, 'alone', 'h2d')}{f(g, 'alone', 'd2h')} |"
              f"{f(g, 'group', 'h2d')}{f(g, 'group', 'd2h')} |"
              f"{f(g, 'all', 'h2d')}{f(g, 'all', 'd2h')}")
    print(" " + "-" * 74)
    if all_agg:
        print(f" 全卡并发聚合: H2D {all_agg['h2d']:.1f} GB/s, "
              f"D2H {all_agg['d2h']:.1f} GB/s")
    print()
    for who, d, alone, grp in grp_notes:
        pct = grp / alone * 100 if alone else 0
        hint = "  ⇒ 上行被共享(瓶颈在交换机上行/CPU 侧)" if pct < 75 else ""
        print(f" 组[{who}] {d}: 单卡 {alone:.1f} → 同组并发每卡 {grp:.1f} ({pct:.0f}%){hint}")
    print("\n 提示: 若 H2D 明显低于理论值,检查 NUMA 亲和(numactl)、IOMMU/ACS、"
          "降速链路与 pinned 内存。")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="GPU PCIe 拓扑与 H2D/D2H 带宽")
    ap.add_argument("--vendor", default="0x10de", help="GPU 厂商 ID, 默认 NVIDIA 0x10de")
    ap.add_argument("--size-mb", type=int, default=256, help="每次拷贝大小(MB)")
    ap.add_argument("--iters", type=int, default=20, help="每次测量的拷贝次数")
    ap.add_argument("--no-bench", action="store_true", help="只显示拓扑")
    ap.add_argument("--no-wake", action="store_true", help="不做链路唤醒")
    args = ap.parse_args()

    if not os.path.isdir(SYSFS):
        sys.exit("找不到 /sys/bus/pci/devices, 需要在 Linux 上运行。")
    gpus = discover_gpus(args.vendor)
    if not gpus:
        sys.exit("未发现 GPU (虚拟机/容器中 sysfs 可能不完整)。")

    torch = None if args.no_bench and args.no_wake else map_torch(gpus)
    keep = None
    if torch and not args.no_wake:
        keep = wake_gpus(torch, gpus)

    names = lspci_names()
    groups = print_topology(gpus, names)
    del keep

    if args.no_bench:
        return
    if torch is None:
        print("未检测到 PyTorch(CUDA), 跳过测速。可 pip install torch, "
              "或使用 nvbandwidth。")
        return
    run_bench(torch, gpus, groups, args.size_mb, args.iters)


if __name__ == "__main__":
    main()

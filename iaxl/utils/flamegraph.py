# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux perf helper that renders folded stacks and a flame graph SVG.

perf samples every thread of the process, including native and kernel frames.
Sampling runs through perf's control FIFO, so only the regions marked with
start()/stop() reach the flame graph.
"""

from __future__ import annotations

import errno
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zlib
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from xml.sax.saxutils import escape

__all__ = ["PerfRecorder"]

_FRAME_HEIGHT = 16
_FONT_SIZE = 11
_CHAR_WIDTH = 0.59 * _FONT_SIZE
_MARGIN_TOP = 32
_MARGIN_BOTTOM = 8

_TIMESTAMP = re.compile(r"\s(\d+\.\d+):")
_ACK_TIMEOUT = 10.0
_EDGE_SLACK = 0.02  # perf enables sampling slightly before start() returns


class _Node:
    __slots__ = ("name", "value", "children")

    def __init__(self, name: str) -> None:
        self.name = name
        self.value = 0
        self.children: dict[str, _Node] = {}

    def child(self, name: str) -> "_Node":
        node = self.children.get(name)
        if node is None:
            node = _Node(name)
            self.children[name] = node
        return node


class PerfRecorder:
    """Records native stacks with `perf record`, limited to marked regions."""

    def __init__(self, *, frequency: int = 999, call_graph: str = "fp") -> None:
        assert shutil.which("perf"), "perf is not installed"
        paranoid = _perf_event_paranoid()
        assert paranoid <= 2, (
            f"kernel.perf_event_paranoid={paranoid}; run "
            "'sysctl -w kernel.perf_event_paranoid=1' or use a privileged container"
        )
        # Python 3.12+ emits perf trampolines so interpreter frames get real names.
        trampoline = getattr(sys, "activate_stack_trampoline", None)
        if trampoline is not None:
            trampoline("perf")

        self._directory = tempfile.mkdtemp(prefix="iaxl-flamegraph-")
        self.data_path = os.path.join(self._directory, "perf.data")
        control_path = os.path.join(self._directory, "control.fifo")
        ack_path = os.path.join(self._directory, "ack.fifo")
        os.mkfifo(control_path)
        os.mkfifo(ack_path)
        self._process = subprocess.Popen(
            [
                "perf",
                "record",
                "--quiet",
                "--clockid",
                "mono",
                "--freq",
                str(frequency),
                "--call-graph",
                call_graph,
                "--pid",
                str(os.getpid()),
                "--output",
                self.data_path,
                "--delay",
                "-1",
                "--control",
                f"fifo:{control_path},{ack_path}",
            ]
        )
        self._control = self._open_control(control_path)
        self._ack = os.open(ack_path, os.O_RDWR | os.O_NONBLOCK)
        self._regions: list[tuple[float, float, tuple[str, ...]]] = []
        self._pending: tuple[float, tuple[str, ...]] | None = None
        self._counts: dict[str, int] | None = None

    @property
    def samples(self) -> int:
        self.close()
        assert self._counts is not None
        return sum(self._counts.values())

    def start(self, label: Iterable[str] = ()) -> None:
        assert self._pending is None, "recorder is already running"
        self._command("enable")
        self._pending = (time.monotonic(), tuple(part.replace(";", ":") for part in label))

    def stop(self) -> None:
        assert self._pending is not None, "recorder is not running"
        end = time.monotonic()
        self._command("disable")
        start, prefix = self._pending
        self._pending = None
        self._regions.append((start, end, prefix))

    def close(self) -> None:
        if self._counts is not None:
            return
        assert self._pending is None, "stop the running region first"
        os.close(self._control)
        os.close(self._ack)
        self._process.send_signal(signal.SIGINT)
        # perf record reports the interrupt in its exit status, so only the data matters.
        self._process.wait(timeout=60)
        assert os.path.exists(self.data_path), f"perf record wrote no {self.data_path}"
        self._counts = self._fold()

    def _open_control(self, path: str) -> int:
        deadline = time.monotonic() + _ACK_TIMEOUT
        while True:
            assert self._process.poll() is None, "perf record exited, see its output above"
            try:
                return os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as error:
                assert error.errno == errno.ENXIO, str(error)
                assert time.monotonic() < deadline, "timed out waiting for perf record"
                time.sleep(0.01)

    def _command(self, name: str) -> None:
        assert self._counts is None, "recorder is already closed"
        os.write(self._control, f"{name}\n".encode())
        ready, _, _ = select.select([self._ack], [], [], _ACK_TIMEOUT)
        assert ready, f"perf did not acknowledge '{name}'"
        os.read(self._ack, 64)

    def _fold(self) -> dict[str, int]:
        script = subprocess.run(
            ["perf", "script", "--input", self.data_path, "--no-inline"],
            capture_output=True,
            text=True,
        )
        assert script.returncode == 0, f"perf script failed: {script.stderr.strip()}"
        starts = [region[0] for region in self._regions]
        counts: defaultdict[str, int] = defaultdict(int)
        comm = "unknown"
        timestamp = 0.0
        stack: list[str] = []

        def flush() -> None:
            if not stack:
                return
            key = ";".join(self._label(starts, timestamp) + (comm,) + tuple(reversed(stack)))
            counts[key] += 1
            stack.clear()

        for line in script.stdout.splitlines():
            if not line.strip():
                flush()
            elif line[0].isspace():
                stack.append(_parse_frame(line))
            else:
                flush()
                comm = line.split()[0]
                match = _TIMESTAMP.search(line)
                timestamp = float(match.group(1)) if match else 0.0
        flush()
        return dict(counts)

    def _label(self, starts: list[float], timestamp: float) -> tuple[str, ...]:
        index = bisect_right(starts, timestamp + _EDGE_SLACK) - 1
        if index < 0:
            return ("unmatched",)
        _, end, prefix = self._regions[index]
        return prefix if timestamp <= end + _EDGE_SLACK else ("unmatched",)

    def write_folded(self, path: str) -> None:
        self.close()
        assert self._counts is not None
        with open(path, "w", encoding="utf-8") as handle:
            for stack, count in sorted(self._counts.items()):
                handle.write(f"{stack} {count}\n")

    def write_svg(self, path: str, title: str = "Flame Graph", width: int = 1400) -> None:
        self.close()
        assert self._counts is not None
        root = _Node("all")
        for stack, count in self._counts.items():
            root.value += count
            node = root
            for name in stack.split(";"):
                node = node.child(name)
                node.value += count
        assert root.value, "no samples were collected"

        depth = _tree_depth(root)
        height = _MARGIN_TOP + _MARGIN_BOTTOM + (depth + 1) * _FRAME_HEIGHT
        scale = width / root.value
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            "<style>g:hover rect{stroke:#000;stroke-width:1}"
            f"text{{font-family:Verdana,sans-serif;font-size:{_FONT_SIZE}px;"
            "fill:#000;pointer-events:none}</style>",
            f'<rect width="{width}" height="{height}" fill="#f8f8f8"/>',
            f'<text x="{width / 2:.1f}" y="20" text-anchor="middle" '
            f'font-size="16">{escape(title)}</text>',
        ]
        _render(root, 0.0, 0, scale, height, root.value, parts)
        parts.append("</svg>")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(parts))


def _perf_event_paranoid() -> int:
    path = "/proc/sys/kernel/perf_event_paranoid"
    if not os.path.exists(path):
        return 3
    with open(path, encoding="utf-8") as handle:
        return int(handle.read().strip())


def _parse_frame(line: str) -> str:
    """Turn a `perf script` callchain line into a symbol name."""
    text = line.strip()
    address, _, remainder = text.partition(" ")
    if remainder and all(char in "0123456789abcdef" for char in address):
        text = remainder.strip()
    dso = ""
    if text.endswith(")"):
        index = text.rfind(" (")
        if index > 0:
            dso = os.path.basename(text[index + 2 : -1])
            text = text[:index]
    offset = text.rfind("+0x")
    if offset > 0:
        text = text[:offset]
    if text in ("[unknown]", "") and dso:
        text = f"[{dso}]"
    return text or "[unknown]"


def _tree_depth(node: _Node) -> int:
    if not node.children:
        return 0
    return 1 + max(_tree_depth(child) for child in node.children.values())


def _render(
    node: _Node,
    x: float,
    depth: int,
    scale: float,
    height: int,
    total: int,
    parts: list[str],
) -> None:
    box_width = node.value * scale
    if box_width < 0.2:
        return
    y = height - _MARGIN_BOTTOM - (depth + 1) * _FRAME_HEIGHT
    percent = 100.0 * node.value / total
    parts.append(
        f"<g><title>{escape(node.name)} ({node.value} samples, {percent:.2f}%)</title>"
        f'<rect x="{x:.2f}" y="{y}" width="{box_width:.2f}" '
        f'height="{_FRAME_HEIGHT - 1}" fill="{_color(node.name)}" rx="1"/>'
        f"{_text(node.name, x, y, box_width)}</g>"
    )
    child_x = x
    for child in sorted(node.children.values(), key=lambda item: item.name):
        _render(child, child_x, depth + 1, scale, height, total, parts)
        child_x += child.value * scale


def _text(name: str, x: float, y: int, box_width: float) -> str:
    capacity = int((box_width - 6) / _CHAR_WIDTH)
    if capacity < 3:
        return ""
    text = name if len(name) <= capacity else name[: capacity - 2] + ".."
    return f'<text x="{x + 3:.2f}" y="{y + _FRAME_HEIGHT - 5}">{escape(text)}</text>'


def _color(name: str) -> str:
    digest = zlib.crc32(name.encode("utf-8"))
    red = 205 + (digest & 0xFF) * 50 // 255
    green = ((digest >> 8) & 0xFF) * 230 // 255
    blue = ((digest >> 16) & 0xFF) * 55 // 255
    return f"rgb({red},{green},{blue})"

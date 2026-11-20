"""生成桌面图标（任务 10.1 / 10.2）。

不依赖 Pillow：直接写 PNG + Vista 风格（PNG 压缩）ICO，构建脚本调用一次即可。
输出：``app/desktop/assets/icon.png``（512）与 ``icon.ico``（256）。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "desktop" / "assets"

BG = (11, 15, 20, 255)
CYAN = (34, 211, 238, 255)
SOFT = (8, 145, 178, 255)
WHITE = (226, 232, 240, 255)

# 归一化骨架点（x, y），与项目 2D 姿势示意一致
JOINTS = {
    "head": (0.50, 0.19),
    "neck": (0.50, 0.29),
    "hip": (0.50, 0.55),
    "l_shoulder": (0.34, 0.33),
    "r_shoulder": (0.66, 0.33),
    "l_elbow": (0.24, 0.46),
    "r_elbow": (0.76, 0.46),
    "l_hand": (0.18, 0.59),
    "r_hand": (0.82, 0.59),
    "l_knee": (0.38, 0.73),
    "r_knee": (0.62, 0.73),
    "l_foot": (0.34, 0.90),
    "r_foot": (0.66, 0.90),
}
BONES = [
    ("head", "neck"),
    ("neck", "hip"),
    ("neck", "l_shoulder"),
    ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_hand"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_hand"),
    ("hip", "l_knee"),
    ("l_knee", "l_foot"),
    ("hip", "r_knee"),
    ("r_knee", "r_foot"),
]


def _canvas(size: int) -> list[list[int]]:
    return [[0, 0, 0, 0] for _ in range(size * size)]


def _blend(canvas: list[list[int]], size: int, x: int, y: int, color: tuple[int, int, int, int]) -> None:
    if not (0 <= x < size and 0 <= y < size):
        return
    r, g, b, a = color
    dst = canvas[y * size + x]
    src_a = a / 255
    dst_a = dst[3] / 255
    out_a = src_a + dst_a * (1 - src_a)
    if out_a <= 0:
        return
    for index, value in enumerate((r, g, b)):
        dst[index] = round((value * src_a + dst[index] * dst_a * (1 - src_a)) / out_a)
    dst[3] = round(out_a * 255)


def _disc(canvas: list[list[int]], size: int, cx: float, cy: float, radius: float, color) -> None:
    inner = (radius - 1.0) ** 2
    outer = (radius + 1.0) ** 2
    for y in range(max(0, int(cy - radius - 1)), min(size, int(cy + radius + 2))):
        for x in range(max(0, int(cx - radius - 1)), min(size, int(cx + radius + 2))):
            distance = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2
            if distance <= inner:
                _blend(canvas, size, x, y, color)
            elif distance <= outer:
                coverage = (outer - distance) / (outer - inner)
                _blend(canvas, size, x, y, (color[0], color[1], color[2], round(color[3] * coverage)))


def _line(canvas: list[list[int]], size: int, a, b, width: float, color) -> None:
    x0, y0 = a
    x1, y1 = b
    steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) * 2))
    for step in range(steps + 1):
        t = step / steps
        _disc(canvas, size, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, width / 2, color)


def _rounded_rect(canvas: list[list[int]], size: int, radius: float, color) -> None:
    for y in range(size):
        for x in range(size):
            dx = max(radius - x - 0.5, 0, x + 0.5 - (size - radius))
            dy = max(radius - y - 0.5, 0, y + 0.5 - (size - radius))
            if dx * dx + dy * dy <= radius * radius:
                _blend(canvas, size, x, y, color)


def render(size: int = 512) -> list[list[int]]:
    supersample = 2 if size <= 256 else 1
    work = size * supersample
    canvas = _canvas(work)
    _rounded_rect(canvas, work, work * 0.22, BG)

    # 柔和底盘
    _disc(canvas, work, work * 0.5, work * 0.56, work * 0.33, (12, 40, 48, 160))

    scale = work
    points = {name: (x * scale, y * scale) for name, (x, y) in JOINTS.items()}
    for start, end in BONES:
        _line(canvas, work, points[start], points[end], work * 0.045, CYAN)
    _disc(canvas, work, *points["head"], work * 0.065, WHITE)
    for name in ("neck", "hip", "l_elbow", "r_elbow", "l_knee", "r_knee"):
        _disc(canvas, work, *points[name], work * 0.026, SOFT)
    for name in ("l_hand", "r_hand", "l_foot", "r_foot"):
        _disc(canvas, work, *points[name], work * 0.030, CYAN)

    if supersample == 1:
        return canvas

    # 简单平均下采样，得到抗锯齿边缘
    result = _canvas(size)
    for y in range(size):
        for x in range(size):
            total = [0, 0, 0, 0]
            for dy in range(supersample):
                for dx in range(supersample):
                    pixel = canvas[(y * supersample + dy) * work + (x * supersample + dx)]
                    for index in range(4):
                        total[index] += pixel[index]
            count = supersample * supersample
            result[y * size + x] = [round(value / count) for value in total]
    return result


def _png_bytes(canvas: list[list[int]], size: int) -> bytes:
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            raw.extend(canvas[y * size + x])

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _ico_bytes(png: bytes, size: int) -> bytes:
    entry = struct.pack(
        "<BBBBHHII",
        0 if size >= 256 else size,
        0 if size >= 256 else size,
        0,
        0,
        1,
        32,
        len(png),
        22,
    )
    return struct.pack("<HHH", 0, 1, 1) + entry + png


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    large = render(512)
    (ASSETS / "icon.png").write_bytes(_png_bytes(large, 512))
    small = render(256)
    (ASSETS / "icon.ico").write_bytes(_ico_bytes(_png_bytes(small, 256), 256))
    print(f"图标已生成：{ASSETS / 'icon.png'}、{ASSETS / 'icon.ico'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

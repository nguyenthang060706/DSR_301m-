"""src/dsr/transforms.py

D3-C3 / Spec 8.8: Module biến đổi bounding box và crop jittering (JitterBoxCrop).
Mô phỏng sai lệch định vị và tỷ lệ của Detector khi suy luận trên khay ăn thực tế.

Quy định:
- JitterBoxCrop(shift=0.10, scale=0.15, pad=(0.05, 0.20)):
  + shift: dịch chuyển tâm box ngẫu nhiên trong khoảng [-shift*w, shift*w], [-shift*h, shift*h].
  + scale: co dãn kích thước box ngẫu nhiên trong khoảng [1 - scale, 1 + scale].
  + pad: mở rộng viền box ngẫu nhiên theo tỷ lệ trong khoảng pad (mặc định 5% đến 20%).
- Tọa độ sau jitter được clamp chặt trong kích thước ảnh [0, img_w], [0, img_h].
- Chỉ áp dụng ở Pha 2 và Few-shot finetune (các run Main baseline không có jitter).
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple, Union


def apply_box_jitter(
    box: Sequence[float],
    img_size: Tuple[int, int],
    shift: float = 0.10,
    scale: float = 0.15,
    pad: Tuple[float, float] = (0.05, 0.20),
    rng: random.Random | None = None,
) -> Tuple[int, int, int, int]:
    """Áp dụng jitter ngẫu nhiên lên box [x1, y1, x2, y2].

    Args:
        box: [x1, y1, x2, y2] pixel coordinates.
        img_size: (img_w, img_h).
        shift: biên độ dịch chuyển tâm (mặc định 0.10 = ±10% kích thước box).
        scale: biên độ co dãn (mặc định 0.15 = 85% - 115%).
        pad: khoảng padding ngẫu nhiên (mặc định 5% - 20%).
        rng: random number generator.

    Returns:
        (x1_new, y1_new, x2_new, y2_new) tọa độ nguyên đã clamp trong ảnh.
    """
    rand = rng or random

    img_w, img_h = img_size
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    # 1. Random Shift tâm
    dx = rand.uniform(-shift, shift) * w
    dy = rand.uniform(-shift, shift) * h
    new_cx = cx + dx
    new_cy = cy + dy

    # 2. Random Scale kích thước
    s = rand.uniform(1.0 - scale, 1.0 + scale)
    new_w = w * s
    new_h = h * s

    # 3. Random Padding viền
    p = rand.uniform(pad[0], pad[1])
    padded_w = new_w * (1.0 + 2.0 * p)
    padded_h = new_h * (1.0 + 2.0 * p)

    # 4. Tính toán tọa độ mới
    nx1 = new_cx - padded_w / 2.0
    ny1 = new_cy - padded_h / 2.0
    nx2 = new_cx + padded_w / 2.0
    ny2 = new_cy + padded_h / 2.0

    # 5. Clamp vào kích thước ảnh
    cx1 = max(0, min(img_w - 1, int(round(nx1))))
    cy1 = max(0, min(img_h - 1, int(round(ny1))))
    cx2 = max(cx1 + 1, min(img_w, int(round(nx2))))
    cy2 = max(cy1 + 1, min(img_h, int(round(ny2))))

    return cx1, cy1, cx2, cy2


class JitterBoxCrop:
    """Callable class thực hiện crop với jitter ngẫu nhiên cho PIL Image."""

    def __init__(
        self,
        shift: float = 0.10,
        scale: float = 0.15,
        pad: Tuple[float, float] = (0.05, 0.20),
        seed: int | None = None,
    ) -> None:
        self.shift = shift
        self.scale = scale
        self.pad = pad
        self.rng = random.Random(seed) if seed is not None else random

    def __call__(self, image, box: Sequence[float]):
        """Crop PIL image tại bounding box đã jitter.

        Args:
            image: PIL.Image
            box: [x1, y1, x2, y2]
        Returns:
            PIL.Image crop
        """
        img_w, img_h = image.size
        jx1, jy1, jx2, jy2 = apply_box_jitter(
            box,
            (img_w, img_h),
            shift=self.shift,
            scale=self.scale,
            pad=self.pad,
            rng=self.rng,
        )
        return image.crop((jx1, jy1, jx2, jy2))

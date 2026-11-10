"""把标注 jsonl 转成 YOLO 训练目录。

仓库里的标注是 jsonl（一张图一行、框用像素坐标），YOLO 要的是
images/ + labels/ + data.yaml。这个脚本做转换和切分。

用法：
    python tools/jsonl_to_yolo.py --jsonl data/live4_boxes.jsonl \
        --images images/live4_4char --out dataset --val-ratio 0.1

产出：
    dataset/
    ├── images/train/*.jpg   images/val/*.jpg
    ├── labels/train/*.txt   labels/val/*.txt
    └── data.yaml

说明：
    - 只转 reviewed=true 的记录（预填没核对的不进训练集）
    - 坐标从像素转成 0~1 归一化的 cx cy w h
    - 按图片划分，同一张图不会同时出现在 train 和 val
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-reviewed", action="store_true", default=True)
    a = ap.parse_args()

    src = Path(a.images)
    out = Path(a.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    recs = []
    for line in Path(a.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if a.min_reviewed and not (r.get("reviewed") or r.get("ts")):
            continue
        boxes = [b for b in r.get("boxes") or [] if (b.get("char") or "").strip()]
        img_path = src / r["img"]
        if not boxes or not img_path.exists():
            continue
        recs.append((r["img"], img_path, boxes))

    if not recs:
        print("没有可用记录（检查 --jsonl / --images 路径，或标注是否 reviewed）")
        return 1

    rng = random.Random(0)
    rng.shuffle(recs)
    n_val = max(1, int(len(recs) * a.val_ratio)) if len(recs) > 5 else 0
    val_names = {name for name, _, _ in recs[:n_val]}

    # 图片尺寸用于归一化
    import cv2

    n_box = 0
    for name, img_path, boxes in recs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        split = "val" if name in val_names else "train"
        shutil.copy2(img_path, out / "images" / split / name)

        lines = []
        for b in boxes:
            x, y, bw, bh = b["x"], b["y"], b["w"], b["h"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            n_box += 1
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    yaml = (f"path: {out.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names: {{0: char}}\n")
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"图片 {len(recs)} 张（train {len(recs)-len(val_names)} / val {len(val_names)}）")
    print(f"框 {n_box} 个")
    print(f"配置 {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

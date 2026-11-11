"""从 jieba 词典和真实语料造三字词语序数据集。

产物三个文件：
    words.txt        过滤后的三字词表
    train.jsonl      训练集，每行一个字符集
    val.jsonl        验证集，按字符集切分，同一组字不会跨集

用法：
    python tools/build_order_corpus.py \
        --jieba-dict jieba_dict.txt \
        --lccc lccc_sample.txt \
        --extra geetest_words.txt \
        --out text_corpus

lccc 给一个大文本文件（一行一句），脚本判断词是否在语料里出现过。
没有语料文件就跳过这一步，只用词频过滤，质量会差一些。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path

DIGITS = re.compile(r"^[一二三四五六七八九十百千万零〇两]+$")
HAS_ASCII = re.compile(r"[A-Za-z0-9]")


def load_jieba(path: Path) -> dict[str, int]:
    """jieba 词典一行是 词 词频 词性，只取三字且词性为名词/动词等实词。"""
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        w, freq = parts[0], parts[1]
        if len(w) != 3 or not freq.isdigit():
            continue
        out[w] = int(freq)
    return out


def keep(word: str, freq: int, min_freq: int) -> bool:
    """四条过滤规则：纯数字、一开头虚词、含字母数字、词频过低。"""
    if DIGITS.match(word):
        return False
    if word.startswith("一") and word[1:] not in ("起", "共", "直", "向", "切", "旦"):
        return False
    if HAS_ASCII.search(word):
        return False
    if freq < min_freq:
        return False
    return True


def seen_in_corpus(words: set[str], corpus: Path, min_hits: int = 1) -> set[str]:
    """把语料读一遍，统计每个词是否真的出现过。"""
    hits: dict[str, int] = {}
    text_parts: list[str] = []
    for line in corpus.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line:
            text_parts.append(line)
    text = "\n".join(text_parts)
    for w in words:
        c = text.count(w)
        if c >= min_hits:
            hits[w] = c
    return set(hits)


def build_samples(words: set[str]) -> list[dict]:
    """一个词拆三个字，六个排列当候选，原词是答案；同组字有两个词的标成多解。"""
    by_charset: dict[tuple[str, ...], set[str]] = {}
    for w in words:
        by_charset.setdefault(tuple(sorted(w)), set()).add(w)

    rows = []
    for charset, answers in by_charset.items():
        cands = ["".join(p) for p in itertools.permutations(charset)]
        cands = list(dict.fromkeys(cands))
        rows.append({
            "chars": list(charset),
            "cands": cands,
            "answers": sorted(answers),
            "ambiguous": len(answers) > 1,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jieba-dict", required=True)
    ap.add_argument("--lccc", default="", help="大文本语料，一行一句；不给就只按词频过滤")
    ap.add_argument("--extra", default="", help="额外词表，一行一个词")
    ap.add_argument("--out", default="text_corpus")
    ap.add_argument("--min-freq", type=int, default=100)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jieba = load_jieba(Path(args.jieba_dict))
    words = {w for w, f in jieba.items() if keep(w, f, args.min_freq)}
    print(f"jieba 三字词 {len(jieba)} 个，过滤后 {len(words)} 个")

    if args.lccc:
        words = seen_in_corpus(words, Path(args.lccc))
        print(f"在语料中出现过的词 {len(words)} 个")

    if args.extra:
        extra = {l.strip() for l in Path(args.extra).read_text(encoding="utf-8").splitlines()
                 if len(l.strip()) == 3}
        print(f"额外词 {len(extra)} 个，去重后合计 {len(words | extra)} 个")
        words |= extra

    rows = build_samples(words)
    amb = sum(1 for r in rows if r["ambiguous"])
    print(f"字符集 {len(rows)} 组，其中多解 {amb} 组")

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    cut = int(len(rows) * args.val_ratio)
    val, train = rows[:cut], rows[cut:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "words.txt").write_text("\n".join(sorted(words)), encoding="utf-8")
    for name, part in (("train.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"训练 {len(train)} 组，验证 {len(val)} 组，写到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
检查所有名为 `clipped` 的目录下的 .srt 文件。
如果文件行数（原始或非空） != 3, 则把该文件路径与行数写入输出 JSON。

示例：
    python check_clipped_srt.py /path/to/root -o /tmp/bad_srt.json
    python check_clipped_srt.py /path/to/root --mode raw   # 统计原始行数（含空行）
    python check_clipped_srt.py /path/to/root --mode nonempty  # 统计非空行（默认）
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


def read_text_try_encodings(path: Path, encodings: Optional[List[str]] = None) -> str:
    if encodings is None:
        encodings = ["utf-8", "gbk"]
    last_exc = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc), enc
        except Exception as e:
            last_exc = e
    raise last_exc


def count_lines(text: str, mode: str = "nonempty") -> int:
    """
    mode:
      - 'raw': 包括所有行（按换行分割）
      - 'nonempty': 忽略空白/空行，统计非空行数
    """
    if mode == "raw":
        return len(text.splitlines())
    else:
        return sum(1 for ln in text.splitlines() if ln.strip())


def find_all_clipped_srt_files(root: Path) -> List[Path]:
    """
    递归查找所有名为 'clipped' 的目录，并返回这些目录下所有 .srt 文件（不递归子目录）。
    """
    srt_files: List[Path] = []
    # for dirpath, dirnames, filenames in os.walk(root):
    walker = os.walk(root)
    for dirpath, _, filenames in tqdm(walker, total=None, desc="walking dirs"):
        cur = Path(dirpath)
        if cur.name.lower() == "clipped":
            for fn in filenames:
                if fn.lower().endswith(".srt"):
                    srt_files.append(cur.joinpath(fn))
    return srt_files


def trim_srt_keep_last_three_nonempty_lines(text: str) -> str:
    """
    从文本中取出非空行，保留最后三个非空行，按原有顺序返回拼接的文本（每行末尾以换行符结尾）。
    """
    nonempty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_three = nonempty_lines[-3:]
    # 写回时每行单独一行，加末尾换行
    return "\n".join(last_three) + ("\n" if last_three else "")


def check_clipped_srts(root: Path, mode: str = "nonempty") -> List[Dict]:
    """
    返回不满足 3 行约束的 .srt 文件列表（每项为 dict: path, raw_lines, nonempty_lines 或 error）
    使用单一总体进度条显示处理进度。
    """
    results: List[Dict] = []
    srt_files = find_all_clipped_srt_files(root)

    if not srt_files:
        return results

    for entry in tqdm(srt_files, desc="Checking .srt files"):
        try:
            text, used_enc = read_text_try_encodings(entry)
        except Exception as e:
            results.append({
                "path": str(entry.resolve()),
                "error": f"read_error: {repr(e)}"
            })
            continue

        raw_count = count_lines(text, mode="raw")
        nonempty_count = count_lines(text, mode="nonempty")
        check_count = raw_count if mode == "raw" else nonempty_count
        if check_count == 3:
            continue
        elif check_count > 3:
            # 截取最后三行非空行并写回
            try:
                new_text = trim_srt_keep_last_three_nonempty_lines(text)
                entry.write_text(new_text, encoding=used_enc)
                results.append({
                    "path": str(entry.resolve()),
                    "raw_lines": raw_count,
                    "nonempty_lines": nonempty_count,
                    "action": "trimmed_to_last_3",
                    "encoding": used_enc
                })
            except Exception as e:
                results.append({
                    "path": str(entry.resolve()),
                    "raw_lines": raw_count,
                    "nonempty_lines": nonempty_count,
                    "error": f"write_error: {repr(e)}",
                    "encoding": used_enc
                })
        else:
            results.append({
                "path": str(entry.resolve()),
                "raw_lines": raw_count,
                "nonempty_lines": nonempty_count,
                "error": "too_few_lines"  # 你要求 0 行报错，我把 <3 都当做错误记录，便于后续筛查
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="检查 clipped 目录下 .srt 文件是否恰好 3 行")
    parser.add_argument("--root", type=str, nargs='?',
                    default="/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh",
                    help="根目录（递归查找名为 clipped 的文件夹）")
    parser.add_argument("-o", "--output", type=str, default="correct_srt.json", help="输出 JSON 文件路径")
    parser.add_argument("--mode", choices=("raw", "nonempty"), default="nonempty",
                        help="统计方式：raw 包含所有行；nonempty 只统计非空行（默认）")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"[ERROR] root 不存在: {root}")

    print(f"[INFO] Searching clipped dirs under: {root}")
    bad_list = check_clipped_srts(root, mode=args.mode)
    out_path = Path(args.output).expanduser().resolve()
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(bad_list, fout, ensure_ascii=False, indent=2)

    print(f"[INFO] Found {len(bad_list)} problematic .srt files. Results written to: {out_path}")

if __name__ == "__main__":
    main()

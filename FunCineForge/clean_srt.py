#!/usr/bin/env python3
import os
import re
import argparse
import unicodedata
import traceback
import contextlib
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import soundfile as sf
from tqdm import tqdm
import json

# 配置
MIN_DUP_SUBSTR_LEN = 3
MAX_DUP_SUBSTR_LEN = 10
DUP_SUBSTR_OCCURS = 5
DUP_SUBSTR_UNIQUE_THRESHOLD = 1
ADJ_REPEAT_MIN_RUN = 5
DEFAULT_MIN_AUDIO_SEC_FOR_TEXT_CHECK = 10
DEFAULT_MIN_CJK_CHARS = 10

# 用于识别索引行（可能带 speaker）和时间戳行
INDEX_RE = re.compile(r'^\s*(\d+)(?:\s+(\S+))?\s*$')
TIMESTAMP_RE = re.compile(r'^\s*\d{2}:\d{2}:\d{2}[,\.]\d+\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d+\s*$', re.M)
ASR_FILLERS = re.compile(r'\b(uh|um|mm|hmm|<unk>|\[noise\]|\[laughter\]|\[inaudible\])\b', re.I)

def is_cjk(char):
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF or
        0x2A700 <= code <= 0x2B73F or
        0x2B740 <= code <= 0x2B81F or
        0x2B820 <= code <= 0x2CEAF or
        0xF900 <= code <= 0xFAFF or
        0x3000 <= code <= 0x303F
    )

def count_char_types(text):
    cjk = ascii_letters = digits = others = 0
    for ch in text:
        if is_cjk(ch):
            cjk += 1
        elif ch.isalpha() and ord(ch) < 128:
            ascii_letters += 1
        elif ch.isdigit():
            digits += 1
        else:
            others += 1
    return {'cjk': cjk, 'ascii': ascii_letters, 'digits': digits, 'others': others}

def clean_srt_text_keep_punct(raw_text):
    """
        - 跳过索引行（支持带 speaker 的索引，例如 "1 spk0"）
        - 跳过时间戳行
        - 删除控制字符、ASR fillers，规范空格
    返回 cleaned_text
    """
    text = raw_text.replace('\ufeff', '')
    text = unicodedata.normalize('NFKC', text)
    lines = []
    for line in text.splitlines():
        ls = line.strip()
        if not ls:
            continue
        # 索引行跳过
        if INDEX_RE.match(ls):
            continue
        # 时间戳行跳过
        if TIMESTAMP_RE.match(ls):
            continue
        clean_line = ASR_FILLERS.sub(' ', line)
        # 删除控制字符
        lines.append(clean_line)
    cleaned = ' '.join(lines)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_wav_duration(path):
    try:
        with sf.SoundFile(path) as f:
            frames = len(f)
            sr = f.samplerate
            if sr <= 0:
                raise RuntimeError("invalid samplerate")
            return frames / sr
    except Exception:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            if sr <= 0:
                raise RuntimeError("invalid framerate")
            return frames / sr

def find_repeated_substrings(text, min_len=MIN_DUP_SUBSTR_LEN, max_len=MAX_DUP_SUBSTR_LEN, min_occurs=DUP_SUBSTR_OCCURS):
    t = text.replace(' ', '')
    n = len(t)
    counts = Counter()
    if n < min_len:
        return {}
    max_len = min(max_len, n)
    for L in range(min_len, max_len + 1):
        seen = {}
        for i in range(0, n - L + 1):
            sub = t[i:i+L]
            seen[sub] = seen.get(sub, 0) + 1
        for sub, c in seen.items():
            if c >= min_occurs:
                counts[sub] = c
    return dict(counts)

def find_adjacent_repeats(text):
    if not text:
        return 0, []
    tokens = text.split()
    repeats = []
    if len(tokens) >= 2:
        i = 0
        while i < len(tokens)-1:
            run_token = tokens[i]
            run_len = 1
            j = i+1
            while j < len(tokens) and tokens[j] == run_token:
                run_len += 1
                j += 1
            if run_len >= ADJ_REPEAT_MIN_RUN:
                repeats.append((run_token, run_len, i))
            i = j
        return len(repeats), repeats
    else:
        s = text.replace(' ', '')
        repeats = []
        i = 0
        while i < len(s)-1:
            ch = s[i]
            j = i+1
            run_len = 1
            while j < len(s) and s[j] == ch:
                run_len += 1
                j += 1
            # 只有当重复长度达到阈值且不是单个字符时才记录
            if run_len >= ADJ_REPEAT_MIN_RUN:
                # 查找整个重复序列
                full_run = s[i:j]
                # 只有当序列中所有字符都相同且长度为1时才忽略
                if not (len(set(full_run)) == 1 and len(full_run) > 1):
                    repeats.append((full_run, run_len, i))
            i = j
        return len(repeats), repeats

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

def trim_srt_keep_last_three_nonempty_lines(text: str) -> str:
    """
    从文本中取出非空行，保留最后三个非空行，按原有顺序返回拼接的文本（每行末尾以换行符结尾）。
    """
    nonempty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_three = nonempty_lines[-3:]
    # 写回时每行单独一行，加末尾换行
    return "\n".join(last_three) + ("\n" if last_three else "")


def process_one_srt(srt_path, min_audio_sec_for_text_check=DEFAULT_MIN_AUDIO_SEC_FOR_TEXT_CHECK, min_cjk_chars=DEFAULT_MIN_CJK_CHARS):
    out = {
        'srt_path': srt_path,
        'wav_duration': 0,
        'cjk_count': 0,
        'ascii_count': 0,
        'repeated_substr_count': 0,
        'repeated_substr_examples': '',
        'adj_repeat_count': 0,
        'adj_repeat_examples': '',
        'flags': [],
        'lines_correct': False
    }
    try:
        # 读原始并清洗，提取 speaker 列表
        with open(srt_path, 'r', encoding='utf-8', errors='strict') as f:
            raw = f.read()
        check_count = count_lines(raw, mode='nonempty')
        if check_count > 3:
            # 截取最后三行非空行并写回
            try:
                new_text = trim_srt_keep_last_three_nonempty_lines(raw)
                with open(srt_path, 'w', encoding='utf-8') as wf:
                    wf.write(new_text)
                out['lines_correct'] = True
            except Exception as e:
                print(f"[ERROR] {srt_path} can't write")
        elif check_count < 3:
            out['flags'].append('too_few_lines')
            
        cleaned = clean_srt_text_keep_punct(raw)
        ct = count_char_types(cleaned)
        out['cjk_count'] = ct['cjk']
        out['ascii_count'] = ct['ascii']

        # 只要出现任意英文字符即判定
        if out['ascii_count'] >= 1:
            out['flags'].append('language_mismatch')

        # 重复子串检测
        dup_subs = find_repeated_substrings(cleaned)
        out['repeated_substr_count'] = len(dup_subs)
        if dup_subs:
            examples = sorted(dup_subs.items(), key=lambda x: -x[1])[:5]
            out['repeated_substr_examples'] = ';'.join(f"{k}({v})" for k,v in examples)
        if len(dup_subs) >= DUP_SUBSTR_UNIQUE_THRESHOLD:
            out['flags'].append('multiple_repeated_substrings')

        # 相邻重复检测
        adj_cnt, adj_ex = find_adjacent_repeats(cleaned)
        out['adj_repeat_count'] = adj_cnt
        if adj_ex:
            out['adj_repeat_examples'] = ';'.join(f"{tok}x{run}@" + str(pos) for tok,run,pos in adj_ex[:5])
            out['flags'].append('adjacent_repeats')
            
        # 查找同名 wav
        dirp = os.path.dirname(srt_path)
        base = os.path.splitext(os.path.basename(srt_path))[0]
        wav_candidate = os.path.join(dirp, base + '.wav')
        out['wav_duration'] = get_wav_duration(wav_candidate)
        # 文本过少 vs 音频过长
        dur = out['wav_duration']
        if dur != 0:
            if dur >= min_audio_sec_for_text_check and out['cjk_count'] <= min_cjk_chars:
                out['flags'].append('too_short_text_for_audio')
        else:
            print(f"[ERROR] {wav_candidate} 音频出错")

    except Exception as e:
        out['flags'].append(f'process_error: {e}')
    return out

def iter_clipped_dirs(root_dir):
    for dirpath, _, _ in os.walk(root_dir):
        if os.path.basename(dirpath) == 'clipped':
            yield dirpath

def iter_srt_paths(root_dir):
    for cdir in iter_clipped_dirs(root_dir):
        try:
            with os.scandir(cdir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith('.srt'):
                        yield entry.path
        except Exception:
            continue

def count_srt_quick(root_dir):
    total = 0
    for cdir in iter_clipped_dirs(root_dir):
        try:
            with os.scandir(cdir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith('.srt'):
                        total += 1
        except Exception:
            continue
    return total

def main(root_dir, workers, max_outstanding, min_audio_sec_for_text_check, min_cjk_chars, execute, delete_log):
    total = count_srt_quick(root_dir)
    
    # 初始化统计计数器
    stats = {
        'total_files': total,
        'with_errors': 0,
        'language_mismatch': 0,
        'multiple_repeated_substrings': 0,
        'adjacent_repeats': 0,
        'too_short_text_for_audio': 0,
        'total_cjk': 0,
        'lines_correct': 0,
        'too_few_lines': 0,
        'total_audio_duration': 0.0,
    }

    srt_iter = iter_srt_paths(root_dir)
    with ThreadPoolExecutor(max_workers=workers) as ex, \
        open(delete_log, 'w', encoding='utf-8') as delf, \
        tqdm(total=total, desc="Checking srt files") as pbar:

        futures = {}
        for _ in range(max_outstanding):
            try:
                p = next(srt_iter)
            except StopIteration:
                break
            futures[ex.submit(process_one_srt, p, min_audio_sec_for_text_check, min_cjk_chars)] = p

        while futures:
            done_iter = as_completed(futures)
            done_fut = next(done_iter)
            src = futures.pop(done_fut)
            try:
                r = done_fut.result()
                flags = r.get('flags', [])
                if flags:
                    stats['with_errors'] += 1
                    # 删除逻辑
                    base_path = os.path.splitext(r['srt_path'])[0]
                    for ext in ['.srt', '.wav', '.mp4']:
                        file_path = base_path + ext
                        if os.path.exists(file_path):
                            if execute:
                                try:
                                    os.remove(file_path)
                                    delf.write(json.dumps(r, ensure_ascii=False) + '\n')
                                except Exception as e:
                                    print(f"[删除失败] {file_path}: {e}")
                            else:
                                delf.write(json.dumps(r, ensure_ascii=False) + '\n')
                

                if r.get('lines_correct'):
                    stats['lines_correct'] += 1
                if 'too_few_lines' in flags:
                    stats['too_few_lines'] += 1
                if 'language_mismatch' in flags:
                    stats['language_mismatch'] += 1
                if 'multiple_repeated_substrings' in flags:
                    stats['multiple_repeated_substrings'] += 1
                if 'adjacent_repeats' in flags:
                    stats['adjacent_repeats'] += 1
                if 'too_short_text_for_audio' in flags:
                    stats['too_short_text_for_audio'] += 1
                
                # 累计字符和时长
                stats['total_cjk'] += r.get('cjk_count', 0)
                if r.get('wav_duration') is not None:
                    stats['total_audio_duration'] += r.get('wav_duration')
                
                
            except Exception as e:
                tb = traceback.format_exc()
                print(f"ERROR processing {src}: {e}\n{tb}\n")
            finally:
                pbar.update(1)

            try:
                nxt = next(srt_iter)
                futures[ex.submit(process_one_srt, nxt, min_audio_sec_for_text_check, min_cjk_chars)] = nxt
            except StopIteration:
                continue

    print(f"结果已保存到 {delete_log}")
    print(f"总文件数: {stats['total_files']}")
    print(f"有问题的文件: {stats['with_errors']} ({stats['with_errors']/stats['total_files']*100:.1f}%)")
    print(f"srt行数纠正: {stats['lines_correct']} ({stats['lines_correct']/stats['total_files']*100:.1f}%)")
    print(f"srt行数少于3行: {stats['too_few_lines']} ({stats['too_few_lines']/stats['total_files']*100:.1f}%)")
    print(f"语言不匹配: {stats['language_mismatch']} ({stats['language_mismatch']/stats['total_files']*100:.1f}%)")
    print(f"重复子串问题: {stats['multiple_repeated_substrings']} ({stats['multiple_repeated_substrings']/stats['total_files']*100:.1f}%)")
    print(f"相邻重复问题: {stats['adjacent_repeats']} ({stats['adjacent_repeats']/stats['total_files']*100:.1f}%)")
    print(f"文本过少: {stats['too_short_text_for_audio']} ({stats['too_short_text_for_audio']/stats['total_files']*100:.1f}%)")
    if stats['total_files'] > 0:
        avg_cjk = stats['total_cjk'] / stats['total_files']
        print(f"平均中文字符/文件: {avg_cjk:.1f}")
        if stats['total_audio_duration'] > 0:
            hours = stats['total_audio_duration'] / 3600
            print(f"总音频时长: {hours:.2f} 小时")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="检查纠正并清洗 srt 文件与对应 wav 的质量")
    ap.add_argument("--root", type=str, nargs='?',
                    default="/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh",
                    help="根目录（递归查找名为 clipped 的文件夹）")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 1) * 4), help="线程数（默认 CPU*4）")
    ap.add_argument("--max_outstanding", type=int, default=max(16, (os.cpu_count() or 1) * 8), help="futures 数量")
    ap.add_argument("--min_audio_sec", type=float, default=DEFAULT_MIN_AUDIO_SEC_FOR_TEXT_CHECK, help="音频足够长以触发文本过少检查的阈值（秒）")
    ap.add_argument("--min_cjk_chars", type=int, default=DEFAULT_MIN_CJK_CHARS, help="文本过少检查的中文字符阈值（<=此数视为过少）")
    ap.add_argument("--execute", action="store_true", help="真正执行删除，默认仅 dry-run，不会实际删除")
    ap.add_argument("--delete_log", default="delete_srt.log", help="记录已被删除的文件")
    args = ap.parse_args()
    main(args.root, args.workers, args.max_outstanding, args.min_audio_sec, args.min_cjk_chars, args.execute, args.delete_log)

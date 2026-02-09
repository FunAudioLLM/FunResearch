#!/usr/bin/env python3
"""
删除小于 min_sec 或 大于 max_sec 的 .wav文件并同时删除同目录下同名的 .mp4 和 .srt。
默认 dry-run 只打印将删除的文件。加 --execute 会真删除。
"""

import os
import argparse
import contextlib
import wave
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf
from tqdm import tqdm

ASSOCIATED_EXTS = ['.wav', '.mp4', '.srt']
EXTRA_DIRS = ['instrumental', 'vocals']

def iter_clipped_dirs(root_dir):
    for dirpath, _, _ in os.walk(root_dir):
        if os.path.basename(dirpath) == 'clipped':
            yield dirpath

def iter_wav_paths(root_dir):
    for cdir in iter_clipped_dirs(root_dir):
        try:
            with os.scandir(cdir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith('.wav'):
                        yield entry.path
        except Exception:
            continue

def count_wavs_quick(root_dir):
    total = 0
    for cdir in iter_clipped_dirs(root_dir):
        try:
            with os.scandir(cdir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith('.wav'):
                        total += 1
        except Exception:
            continue
    return total

def get_duration(path):
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

def find_case_insensitive_file(dirpath, target_name):
    try:
        for entry in os.listdir(dirpath):
            if entry.lower() == target_name.lower():
                return os.path.join(dirpath, entry)
    except Exception:
        return None
    return None

def remove_file(path, execute):
    """如果 execute 为 True 则删除文件并返回 True。否则仅返回 False"""
    if not path:
        return False, "not_found"
    if not execute:
        return True, "dry_run"  # 表示将删除但未实际删除
    try:
        os.remove(path)
        return True, "deleted"
    except Exception as e:
        return False, f"err:{e}"

def process_and_maybe_delete(path, min_sec, max_sec, execute):
    """
    读取 path 时长并判断是否需要删除。返回 dict 包含:
      'wav': path,
      'duration': float or None,
      'to_delete': bool,
      'deleted': dict of ext -> (success_bool, reason)
    """
    res = {'wav': path, 'duration': None, 'to_delete': False, 'deleted': {}}
    try:
        duration = get_duration(path)
        res['duration'] = duration
        # 删除条件： duration < min_sec OR duration > max_sec
        if duration < min_sec or duration > max_sec:
            res['to_delete'] = True
            dirpath = os.path.dirname(path)
            basename = os.path.splitext(os.path.basename(path))[0]
            # 删除 clipped 目录下关联文件（.wav, .mp4, .srt）
            for ext in ASSOCIATED_EXTS:
                target = basename + ext
                found = find_case_insensitive_file(dirpath, target)
                success, reason = remove_file(found, execute)
                res['deleted'][ext] = (found, success, reason)
            # 删除同级 instrumental/vocals 目录下同名 .wav
            parent_dir = os.path.dirname(dirpath)  # clipped 的父目录
            for extra_dir_name in EXTRA_DIRS:
                extra_dir = os.path.join(parent_dir, extra_dir_name)
                if os.path.isdir(extra_dir):
                    target_wav = basename + '.wav'
                    found_extra = find_case_insensitive_file(extra_dir, target_wav)
                    success, reason = remove_file(found_extra, execute)
                    res['deleted'][extra_dir_name] = (found_extra, success, reason)
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    return res

def main(root_dir, min_sec, max_sec, workers, max_outstanding, execute, log_path):
    total_files = count_wavs_quick(root_dir)
    if total_files == 0:
        print("未找到任何 wav 文件（在名为 'clipped' 的目录中）。")
        return

    deleted_summary = {ext: 0 for ext in (ASSOCIATED_EXTS + EXTRA_DIRS)}
    will_delete_count = 0
    processed = 0

    wav_iter = iter_wav_paths(root_dir)

    with ThreadPoolExecutor(max_workers=workers) as ex, \
         open(log_path, 'w', encoding='utf-8') as logf, \
         tqdm(total=total_files, desc="Scanning wavs") as pbar:

        futures = dict()
        # 初始填充
        for _ in range(max_outstanding):
            try:
                p = next(wav_iter)
            except StopIteration:
                break
            futures[ex.submit(process_and_maybe_delete, p, min_sec, max_sec, execute)] = p

        # 动态提交与处理
        while futures:
            done_iter = as_completed(futures)
            done_fut = next(done_iter)
            src_path = futures.pop(done_fut)
            try:
                r = done_fut.result()
                processed += 1
                if r.get('to_delete'):
                    will_delete_count += 1
                    # 记录每种扩展的删除状态
                    for ext, (found, success, reason) in r['deleted'].items():
                        if success and reason in ('deleted', 'dry_run'):
                            deleted_summary[ext] += 1
                        # 写日志：删除或将删除
                        logf.write(f"{found}\t{r['duration']}\t{success}\t{reason}\n")
                    logf.flush()
                # 如果出错则写失败日志
                if 'error' in r:
                    print(r['wav'], r['error'])
            except Exception as e:
                tb = traceback.format_exc()
                print(src_path, str(e))
            finally:
                pbar.update(1)

            # 补充提交以维持并发度
            try:
                next_path = next(wav_iter)
                futures[ex.submit(process_and_maybe_delete, next_path, min_sec, max_sec, execute)] = next_path
            except StopIteration:
                continue

    print("=== 完成 ===")
    print(f"扫描到 wav 总数: {total_files}")
    print(f"处理成功数: {processed}")
    print(f"符合删除条件的 wav 数量: {will_delete_count}")
    print("各扩展被删除（或将被删除）计数：")
    for ext, cnt in deleted_summary.items():
        print(f"    {ext}: {cnt}")
    if not execute:
        print("当前为 DRY-RUN 模式（未实际删除）。要执行删除请加参数 --execute")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="删除小于 min_sec 或 大于 max_sec 的 wav 及其同名 mp4/srt 文件")
    ap.add_argument("--root", type=str, nargs='?',
                    default="/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh",
                    help="根目录（递归查找名为 clipped 的文件夹）")
    ap.add_argument("--min_sec", type=float, default=2.0, help="删除条件：小于此秒数默认2.0")
    ap.add_argument("--max_sec", type=float, default=60.0, help="删除条件：大于此秒数默认60.0")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 1) * 2),
                    help=f"读取并发线程数")
    ap.add_argument("--max-outstanding", type=int, default=os.cpu_count() * 4,
                    help="futures 数量限制")
    ap.add_argument("--execute", action="store_true",
                    help="确认执行删除")
    ap.add_argument("--log", default="delete_video.log", help="记录已被删除的文件")
    args = ap.parse_args()

    main(args.root, args.min_sec, args.max_sec, args.workers, args.max_outstanding, args.execute, args.log)

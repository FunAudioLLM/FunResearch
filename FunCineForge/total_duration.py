#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
import os, struct, wave, json
from tqdm import tqdm

def wav_duration_wave(path):
    try:
        with wave.open(path, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        return None

def wav_duration_riff(path):
    try:
        with open(path, 'rb') as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] != b'RIFF' or header[8:12] != b'WAVE':
                return None
            byte_rate = None
            data_size = None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                chunk_id = hdr[:4]
                chunk_size = struct.unpack('<I', hdr[4:])[0]
                if chunk_id == b'fmt ':
                    fmt = f.read(chunk_size)
                    if len(fmt) >= 12:
                        # audio_format, channels, sample_rate, byte_rate
                        audio_format, channels, sample_rate, byte_rate = struct.unpack('<HHII', fmt[:12])
                    else:
                        f.seek(chunk_size, os.SEEK_CUR)
                elif chunk_id == b'data':
                    data_size = chunk_size
                    break
                else:
                    f.seek(chunk_size, os.SEEK_CUR)
                if chunk_size % 2 == 1:
                    f.seek(1, os.SEEK_CUR)
            if byte_rate and data_size:
                return data_size / float(byte_rate)
    except Exception:
        return None
    return None

def wav_duration(path):
    d = wav_duration_wave(path)
    if d is not None:
        return d
    d = wav_duration_riff(path)
    if d is not None:
        return d
    return 0.0

def collect_wav_files(root_dir):
    wav_files = []
    stack = [root_dir]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith('.wav'):
                            wav_files.append(entry.path)
                    except PermissionError:
                        continue
        except (FileNotFoundError, NotADirectoryError):
            continue
    return wav_files

def total_duration(wav_files, workers=None, cache_file=None):
    """并行计算总时长，使用 executor.map + chunksize 来提升吞吐量。"""
    if workers is None:
        workers = min(32, max(4, (os.cpu_count() or 1) * 2))

    cache = {}
    if cache_file and os.path.isfile(cache_file):
        try:
            with open(cache_file, 'r') as jf:
                cache = json.load(jf)
        except Exception:
            cache = {}

    total = 0.0
    to_process = []
    for p in wav_files:
        if p in cache:
            total += cache[p]
        else:
            to_process.append(p)

    if to_process:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for path, dur in tqdm(zip(to_process, ex.map(wav_duration, to_process, chunksize=16)),
                                  total=len(to_process), desc="统计进度"):
                if dur is None:
                    dur = 0.0
                total += dur
                if cache_file is not None:
                    cache[path] = dur

    return total

if __name__ == "__main__":
    root_dir = '/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh'
    wav_files = collect_wav_files(root_dir)
    print(f"总共找到 {len(wav_files)} 个 WAV 文件")
    total = total_duration(wav_files, workers=None)
    print(f"总时长: {total:.2f} 秒 ({total/3600:.2f} 小时)")

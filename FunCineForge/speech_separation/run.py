import argparse
import yaml
import numpy as np
import time
from ml_collections import ConfigDict
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
import multiprocessing as mp
from typing import List
from utils import demix_track, get_model_from_config

import warnings
warnings.filterwarnings("ignore")

def find_clipped_wavs(root: str) -> List[str]:
    """递归查找 root 下所有 clipped 文件夹内的 wav 文件，
    并且保证这些 wav 文件还没有被人声乐声分离过。"""
    pattern = os.path.join(root, '**', 'clipped', '*.wav')
    files = glob.glob(pattern, recursive=True)
    result = []

    for path in files:
        clipped_dir = os.path.dirname(path)
        parent_dir = os.path.dirname(clipped_dir.rstrip("/"))

        vocals_dir = os.path.join(parent_dir, "vocals")
        instrumental_dir = os.path.join(parent_dir, "instrumental")
        base_name = os.path.basename(path)
        vocals_path = os.path.join(vocals_dir, base_name)
        instrumental_path = os.path.join(instrumental_dir, base_name)
        # 防止重复处理
        if not (os.path.exists(vocals_path) and os.path.exists(instrumental_path)):
            result.append(path)
        else:
            print(f"Skip {path} it has already been processed .")
    return result


def safe_makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # 在并发写入目录时可能出现 race 条件，忽略即可
        pass
    
def process_track_list(device_id: int,
                       model_type: str,
                       config_path: str,
                       model_path: str,
                       track_paths: List[str],
                       use_amp: bool = True):
    """
    每个进程在指定 device_id 上加载模型并处理分配的 track_paths。
    这个函数会在子进程中运行。
    """
    # 载入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    # 初始化模型
    model = get_model_from_config(model_type, config)
    if model_path:
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt)
    torch.backends.cudnn.benchmark = True

    # 选择设备
    if torch.cuda.is_available() and device_id is not None:
        device = torch.device(f'cuda:{device_id}')
        model = model.to(device)
        model_device_str = f'cuda:{device_id}'
    else:
        device = torch.device('cpu')
        model = model.to(device)
        model_device_str = 'cpu'

    model.eval()
    print(f"[PID {os.getpid()}] Loaded model on {model_device_str}, will process {len(track_paths)} tracks.")

    # per-process variables
    first_chunk_time = None

    # loop tracks
    for idx, path in enumerate(track_paths, 1):
        try:
            t0 = time.time()
            # 读取混音
            mix_original, sr = sf.read(path, dtype='float32')
            if mix_original.ndim == 1:
                mix_np = np.stack([mix_original, mix_original], axis=-1)
            else:
                mix_np = mix_original  # shape (N, 2)
            # mixture tensor shape expected: [channels, samples]
            mixture = torch.tensor(mix_np.T, dtype=torch.float32, device=device)

            # demix
            with torch.no_grad():
                if use_amp and device.type == 'cuda':
                    # mixed precision
                    from torch.cuda.amp import autocast
                    with autocast():
                        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)
                else:
                    res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

            # 保存输出：按照 clipped 上一级目录创建 vocals/ instrumental/
            clipped_dir = os.path.dirname(path)
            parent_dir = os.path.dirname(clipped_dir.rstrip("/"))
            vocals_dir = os.path.join(parent_dir, "vocals")
            instrumental_dir = os.path.join(parent_dir, "instrumental")
            safe_makedirs(vocals_dir)
            safe_makedirs(instrumental_dir)

            # 提取 vocals
            vocals_output = res['vocals'].T
            if isinstance(vocals_output, torch.Tensor):
                vocals_output = vocals_output.cpu().numpy()

            # 转单声道（按原脚本）
            if vocals_output.ndim == 2:
                vocals_output = vocals_output.mean(axis=1)

            # write vocals
            vocals_path = os.path.join(vocals_dir, os.path.basename(path))
            sf.write(vocals_path, vocals_output, sr, subtype='PCM_16')

            # 保证 original mix 是单声道用于减法
            if mix_original.ndim == 2:
                mix_mono = mix_original.mean(axis=1)
            else:
                mix_mono = mix_original
            # vocals_output 可能和 mix_mono 长度相同
            instrumental = mix_mono - vocals_output
            instrumental_path = os.path.join(instrumental_dir, os.path.basename(path))
            sf.write(instrumental_path, instrumental, sr, subtype='PCM_16')

            elapsed = time.time() - t0
            print(f"[PID {os.getpid()}] ({idx}/{len(track_paths)}) {os.path.basename(path)} done in {elapsed:.2f}s")
        except Exception as e:
            print(f"[PID {os.getpid()}] Error processing {path}: {e}", file=sys.stderr)

    print(f"[PID {os.getpid()}] Finished processing assigned tracks.")    


def chunk_list(lst, n):
    """把列表 lst 划分为 n 份（尽量平均）"""
    if n <= 0:
        return [lst]
    k, m = divmod(len(lst), n)
    res = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        res.append(lst[start:start+size])
        start += size
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, default='mel_band_roformer')
    p.add_argument("--config_path", type=str, default='configs/config_vocals_mel_band_roformer.yaml', help="path to config yaml file")
    p.add_argument("--model_path", type=str, default='models/melbandroformer/MelBandRoformer.ckpt', help="Location of the model")
    p.add_argument("--root", type=str, required=True, help="root folder to search for 'clipped' subfolders")
    p.add_argument("--gpus", nargs='+', type=int, default=None, help='list of gpu ids. If omitted, runs on CPU (slow).')
    p.add_argument("--use_amp", action='store_true', help="use mixed precision")
    p.add_argument("--max_procs", type=int, default=None, help="max processes to spawn (default = len(device) or number of CPU cores)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 收集所有 wav
    all_wavs = find_clipped_wavs(args.root)
    if len(all_wavs) == 0:
        print("No .wav files found under clipped. Check your input root.")
        sys.exit(1)
    print(f"Found {len(all_wavs)} wav files to process.")

    # 分配任务
    if args.gpus is not None and torch.cuda.is_available():
        device = args.gpus
        n_procs = len(device) if args.max_procs is None else min(len(device), args.max_procs)
        assignments = chunk_list(all_wavs, n_procs)
        procs = []
        mp.set_start_method('spawn', force=True)
        for i in range(n_procs):
            dev = device[i] if i < len(device) else device[0]
            p = mp.Process(target=process_track_list, args=(
                dev, args.model_type, args.config_path, args.model_path, assignments[i], args.use_amp))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        cpu_count = mp.cpu_count()
        n_procs = args.max_procs if args.max_procs is not None else min(len(all_wavs), max(1, cpu_count))
        print(f"CUDA not available or device not given. Running on CPU with {n_procs} processes (may be slow).")
        assignments = chunk_list(all_wavs, n_procs)
        procs = []
        mp.set_start_method('spawn', force=True)
        for i in range(n_procs):
            p = mp.Process(target=process_track_list, args=(
                None, args.model_type, args.config_path, args.model_path, assignments[i], False))  # CPU 不用 amp
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print("All done!")

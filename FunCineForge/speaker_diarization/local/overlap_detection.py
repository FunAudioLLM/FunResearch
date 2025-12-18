"""
This script will download the pretrained models from pyannote/segmentation-3.0 (https://huggingface.co/pyannote/segmentation-3.0)
and perform the overlap detection given the audio. Please pre-install "pyannote".
"""

import os
import numpy as np
import argparse
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import gc
from multiprocessing import get_context
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



def init_worker(hf_access_token, use_gpu, batch_size):
    global _MODEL, _INFERENCE, _DEVICE
    from pyannote.audio import Inference, Model

    device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    _DEVICE = device

    _MODEL = Model.from_pretrained(
        'pyannote/segmentation-3.0',
        use_auth_token=hf_access_token,
        strict=False,
    )

    _INFERENCE = Inference(
        _MODEL,
        duration=_MODEL.specifications.duration,
        step=0.1 * _MODEL.specifications.duration,
        skip_aggregation=True,
        batch_size=batch_size,
        device=device,
    )

    
def _detect_overlap_from_count(count_obj, min_duration=0.5):
    """
    输入 count_obj (SlidingWindowFeature)，返回 overlap intervals 及总时长
    """
    if count_obj is None:
        return [], 0.0

    sliding_window = count_obj.sliding_window
    # duration and step
    # frame_duration = sliding_window.duration
    # frame_step = sliding_window.step

    # Ensure 1D
    count_data = count_obj.data.squeeze()
    overlap_intervals = []
    current_start = None
    total_overlap_duration = 0.0

    # Use sliding_window indexing
    for i, val in enumerate(count_data):
        timestamp = sliding_window[i].start
        if val >= 2:
            if current_start is None:
                current_start = timestamp
        else:
            if current_start is not None:
                current_end = timestamp
                duration = current_end - current_start
                if duration >= min_duration:
                    overlap_intervals.append([current_start, current_end])
                    total_overlap_duration += duration
                current_start = None

    # tail
    if current_start is not None:
        current_end = sliding_window[-1].end
        duration = current_end - current_start
        if duration >= min_duration:
            overlap_intervals.append([current_start, current_end])
            total_overlap_duration += duration

    return overlap_intervals, total_overlap_duration

def process_one(wpath, overlap_threshold):
    """
    在 worker 中调用的单个文件处理函数。
    使用全局的 _INFERENCE（由 init_worker 创建）。
    返回 (wpath, overlap_intervals, total_overlap_duration, error_str_or_None)
    """
    global _INFERENCE, _MODEL, _DEVICE
    try:
        # ensure imports available
        from pyannote.audio import Inference
        # run segmentation
        segmentations = _INFERENCE({'audio': wpath})
        # original code used: frame_windows = _segmentation.model.receptive_field
        frame_windows = _INFERENCE.model.receptive_field
    
        count = Inference.aggregate(
                np.sum(segmentations, axis=-1, keepdims=True),
                frame_windows,
                hamming=False,
                missing=0.0,
                skip_average=False,
            )
        # convert to integer counts
        count.data = np.rint(count.data).astype(np.uint8)

        # detect overlaps
        overlap_intervals, total_overlap_duration = _detect_overlap_from_count(
            count, min_duration=overlap_threshold
        )

        # clean big objects and free memory
        try:
            del segmentations, counts_array
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available() and _DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return (wpath, overlap_intervals, float(total_overlap_duration), None)

    except Exception as e:
        # ensure any big objects cleaned
        try:
            del segmentations
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available() and 'torch' in globals() and _DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        return (wpath, [], 0.0, str(e))


def parse_wavs_arg(wavs_arg):
    wavs = []
    if wavs_arg.endswith('.wav'):
        wavs.append(wavs_arg)
    else:
        # assume it's a file listing wav paths
        with open(wavs_arg, 'r') as f:
            for l in f:
                l = l.strip()
                if l:
                    wavs.append(l)
    return wavs

def main():
    parser = argparse.ArgumentParser(description='Overlap detection (parallel)')
    parser.add_argument('--wavs', required=True, type=str, help='Single wav or path to wav-list file')
    parser.add_argument('--out_dir', required=True, type=str, help='Output dir')
    parser.add_argument('--hf_access_token', default='', type=str, help='HuggingFace access token (if required)')
    parser.add_argument('--overlap_threshold', type=float, default=0.5, help='Minimum overlap duration (s)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: cpu_count)')
    parser.add_argument('--batch_size', type=int, default=32, help='Inference batch size per worker')
    parser.add_argument('--use_gpu', action='store_true', help='Force use GPU in workers (dangerous for multiple workers)')
    parser.add_argument('--nj', type=int, default='0', help='GPUs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    overlap_list_path = os.path.join(args.out_dir, 'overlap.list')
    if os.path.exists(overlap_list_path):
        print(f"[INFO] Overlap list already exists at {overlap_list_path}. Skip.")
        return

    wavs = parse_wavs_arg(args.wavs)
    if not wavs:
        raise Exception("No wav files found.")

    # determine number of workers
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    workers = args.workers if args.workers and args.workers > 0 else cpu_count*2
    # limit workers to number of files
    workers = min(workers, max(1, len(wavs)))
    if args.use_gpu:
        workers = min(workers, args.nj)  # 默认一张卡一个worker

    print(f"[INFO] Overlap detection will process {len(wavs)} files using {workers} workers, GPUs: {args.nj if args.use_gpu else 'None'}, CPUs: {cpu_count}")

    ctx = get_context('spawn')
    # 创建一个带worker_id的init函数包装器
    init_args = (args.hf_access_token, args.use_gpu, args.batch_size)

    # Start pool
    results = []
    try:
        with ctx.Pool(processes=workers, initializer=init_worker, initargs=init_args) as pool:
            with tqdm(total=len(wavs), desc="Processing", unit="file") as pbar:
            # imap_unordered yields results as they complete
                for i, res in enumerate(pool.imap_unordered(partial(process_one, overlap_threshold=args.overlap_threshold), wavs)):
                    results.append(res)
                    wpath, overlaps, total_dur, err = res
                    if err:
                        print(f"[ERROR] {wpath} -> {err}")
                    pbar.update(1)
    except Exception as e:
        print(f"[FATAL] Pool error: {e}")
        raise

    # Collect files with overlap and write list
    overlap_files = [w for (w, o, d, err) in results if err is None and d > 0]

    with open(overlap_list_path, 'w') as f:
        for p in overlap_files:
            f.write(p + '\n')
    print(f"\n[SUCCESS] Found {len(overlap_files)} files with speaker overlap -> {overlap_list_path}")


if __name__ == '__main__':
    main()

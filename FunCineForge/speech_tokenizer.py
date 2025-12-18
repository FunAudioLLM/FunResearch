#!/usr/bin/env python3
import argparse
import os
import onnxruntime
import numpy as np
import torchaudio
import whisper
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_vocal_files_parallel(root_dir: str):
    wav_paths = []
    def walk_dir(dirname):
        paths = []
        for dirpath, dirnames, filenames in os.walk(dirname):
            if os.path.basename(dirpath).lower() == "vocals":
                paths.extend(os.path.join(dirpath, fn) for fn in filenames if fn.endswith(".wav"))
                dirnames[:] = []  # 阻止递归
        return paths

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        # 假设 root_dir 的直接子目录为独立项目（按需调整）
        for d in os.listdir(root_dir):
            full_path = os.path.join(root_dir, d)
            if os.path.isdir(full_path):
                futures.append(executor.submit(walk_dir, full_path))
        for future in tqdm(as_completed(futures), desc="Finding files"):
            wav_paths.extend(future.result())
    return wav_paths

def single_job(wav_path: str):
    audio, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    if audio.shape[0] != 1:
        print(f"[WARNING] Audio {wav_path} is not a single channel, shape is {audio.shape}")
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] / 16000 > 60:
        print(f"[ERROR] Do not support extract speech token for audio longer than 60s")
        return wav_path, np.array([], dtype=np.int32)
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        try:
            speech_token = ort_session.run(None, 
                                    {ort_session.get_inputs()[0].name: 
                                    feat.detach().cpu().numpy(),
                                    ort_session.get_inputs()[1].name: 
                                    np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
            speech_token = np.array(speech_token, dtype=np.int32)
            return wav_path, speech_token
        except Exception as e:
            print(f"[ERROR] Extract speech token failed for {wav_path}")
            return wav_path, np.array([], dtype=np.int32)

def batch_process(wav_paths: List[str]):
    futures = []
    for wpath in tqdm(wav_paths, desc="Submitting jobs"):
        tokens_dir = os.path.join(os.path.dirname(os.path.dirname(wpath)), 'tokens')
        os.makedirs(tokens_dir, exist_ok=True)
        token_filename = os.path.splitext(os.path.basename(wpath))[0] + '.npy'
        token_path = os.path.join(tokens_dir, token_filename)
        if os.path.exists(token_path):
            print(f"[INFO] Speech token already exists for {wpath}, skipping.")
        else:
            futures.append(executor.submit(single_job, wpath))
    for future in tqdm(as_completed(futures), desc="Processing files"):
        try:
            wav_path, speech_token = future.result()
            # tokens save path
            tokens_dir = os.path.join(os.path.dirname(os.path.dirname(wav_path)), 'tokens')
            token_filename = os.path.splitext(os.path.basename(wav_path))[0] + '.npy'
            token_path = os.path.join(tokens_dir, token_filename)
            if speech_token.size > 0:
                np.save(token_path, speech_token)
            else:
                print(f"[WARNING] Empty speech token for {wav_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process job: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="root path")
    parser.add_argument("--onnx_path", type=str, default="tokenizer/speech_tokenizer_v3.onnx", help="onnx model path")
    parser.add_argument("--num_thread", type=int, default=min(os.cpu_count(), 32))
    args = parser.parse_args()
    print(f"[INFO] Beginning...")
    wav_paths = find_vocal_files_parallel(args.root)
    print(f"[INFO] Found {len(wav_paths)} audio files to process")
    
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=["CPUExecutionProvider"])
    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    batch_process(wav_paths)
    print(f"[INFO] All Done!")
#!/usr/bin/env python3
"""
This script normalizes video files by converting them to MP4 format using ffmpeg.
It processes all video files in a specified directory and its subdirectories,
renaming them to a normalized format.
"""
import os
import re
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

VIDEO_EXTENSIONS = ['.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm']

def normalize_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name + ".mp4"

def convert_to_mp4(original_path, converted_path):
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", original_path,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-strict", "experimental",
        converted_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    success = result.returncode == 0
    return original_path, converted_path, success, result.stderr

def collect_video_tasks(root_dir):
    tasks = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                original_path = os.path.join(dirpath, filename)
                normalized_name = normalize_filename(filename)
                converted_path = os.path.join(dirpath, normalized_name)
                if original_path != converted_path:
                    tasks.append((original_path, converted_path))
            elif filename.lower().endswith('mp4'):
                normalized_name = normalize_filename(filename)
                if filename == normalized_name:
                    continue
                original_path = os.path.join(dirpath, filename)
                rename_path = os.path.join(dirpath, normalized_name)
                os.rename(original_path, rename_path)
            else:
                print(f"[WARNING] {filename} is not a suitable file type")
    return tasks

def parallel_convert_all_videos_to_mp4(root_dir, max_workers=8):
    tasks = collect_video_tasks(root_dir)
    print(f"Found {len(tasks)} files to convert.")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_to_mp4, orig, conv) for orig, conv in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting videos"):
            original_path, converted_path, success, err = future.result()
            if success:
                print(f"✅ Converted {original_path} to {converted_path}")
                if original_path != converted_path:
                    try:
                        os.remove(original_path)
                        print(f"Deleted original file {original_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete {original_path}: {e}")
            else:
                print(f"❌ Error converting {original_path}:\n{err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量转码视频为 mp4")
    parser.add_argument("--root", type=str, required=True, help="视频所在的根目录")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="并行线程数")

    args = parser.parse_args()
    parallel_convert_all_videos_to_mp4(args.root, max_workers=args.max_workers)

#!/usr/bin/env python3
import subprocess
import os
import argparse
import multiprocessing as mp
from typing import List

def ensure_dependencies():
    """检查 ffprobe 和 ffmpeg 是否可用"""
    for cmd in ["ffprobe", "ffmpeg"]:
        result = subprocess.run(["which", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERROR] 系统未安装 '{cmd}'，请先安装 ffmpeg 工具包。")
            exit(1)

def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        stdout=subprocess.PIPE,
        text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise ValueError(f"[REEOR] 无法解析时长: {result.stdout}")
    

def trim_video_and_extract_audio(input_path, output_video_path, output_audio_path, intro_sec, outro_sec):
    try:
        if intro_sec == 0 and outro_sec == 0:
            print(f"[VIDEO] 跳过剪切直接提取音频")
            source_for_audio = input_path
        else:
            duration = get_video_duration(input_path)
            start = intro_sec
            end = duration - outro_sec
            if end <= start:
                raise ValueError(f"[REEOR] 视频太短(start={start}, end={end})，无法剪裁")

            video_cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(start),
                "-to", str(end),
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                output_video_path
            ]
            subprocess.run(video_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[VIDEO] {input_path} → {output_video_path}")
            source_for_audio = output_video_path
            
        audio_cmd = [
            "ffmpeg", "-y",
            "-i", source_for_audio,
            "-f", "wav",
            "-vn",  # 只要音频
            "-c:a", "pcm_s16le",
            output_audio_path
        ]
        subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[AUDIO] {input_path} → {output_audio_path}")
        return True
    
    except Exception as e:
        print(f"[ERROR] {input_path}: {e}")
        return False


def collect_videos(root_dir):
    grouped = {}
    for dirpath, _, filenames in os.walk(root_dir):
        mp4s = [os.path.join(dirpath, fn) for fn in filenames if fn.lower().endswith('.mp4')]
        if mp4s:
            grouped[dirpath] = sorted(mp4s)
    return grouped

def process_directory(args):
    vdir, file_list, intro_sec, outro_sec = args
    success_count = 0
    total = len(file_list)

    for idx, input_path in enumerate(file_list, start=1):
        output_video_path = os.path.join(vdir, f"{idx:02d}.mp4")
        output_audio_path = os.path.join(vdir, f"{idx:02d}.wav")

        need_temp = os.path.abspath(input_path) == os.path.abspath(output_video_path)
        if need_temp:
            temp_video = os.path.join(vdir, f"temp_{idx:02d}.mp4")
            temp_audio = os.path.join(vdir, f"temp_{idx:02d}.wav")
        else:
            temp_video, temp_audio = output_video_path, output_audio_path

        success = trim_video_and_extract_audio(
            input_path, temp_video, temp_audio, intro_sec, outro_sec
        )
        if success:
            os.remove(input_path)
            if need_temp:
                os.replace(temp_video, output_video_path)
                os.replace(temp_audio, output_audio_path)
            success_count += 1
    return success_count, total

def batch_trim_videos(input_dir, max_workers, intro_sec=0, outro_sec=0):
    grouped = collect_videos(input_dir)

    tasks = [(vdir, grouped[vdir], intro_sec, outro_sec) for vdir in sorted(grouped.keys())]
    num_workers = min(len(tasks), mp.cpu_count(), max_workers)
    print(f"[INFO] 启动 {num_workers} 个并发 Worker处理 {len(tasks)} 个目录...")
    with mp.Pool(processes=num_workers, initializer=ensure_dependencies) as pool:
        results = pool.map(process_directory, tasks)
    
    total_files = sum(r[1] for r in results)
    total_success = sum(r[0] for r in results)
    print(f"[SUCCESS] {total_success}/{total_files} 个文件")
    


if __name__ == "__main__":
    ensure_dependencies()
    parser = argparse.ArgumentParser(description="批量裁剪视频片头片尾，并提取对应音频")
    parser.add_argument("--root", type=str, required=True, help="视频所在的根目录")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="最大并发 Worker 数")
    parser.add_argument("--intro_sec", type=int, default=5, help="片头裁剪秒数")
    parser.add_argument("--outro_sec", type=int, default=5, help="片尾裁剪秒数")
    args = parser.parse_args()

    batch_trim_videos(args.root, args.workers, args.intro_sec, args.outro_sec)
    print("✅ All videos trim processed.")

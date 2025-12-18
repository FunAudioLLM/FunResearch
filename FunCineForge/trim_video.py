#!/usr/bin/env python3
"""
This script trims videos by removing intro and outro sections.
It processes all video files in a specified directory, trimming them to a specified duration.
"""
import subprocess
import os
import re
import argparse

def get_video_duration(video_path):
    """ffprobe 获取视频总时长"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    output = result.stdout.decode().strip()
    # 提取最后一个浮点数
    match = re.findall(r"\d+\.\d+", output)
    if match:
        return float(match[-1])  # 返回最后一个浮点数
    else:
        raise ValueError(f"无法从输出中提取浮点数: {output}")
    

def trim_video(input_path, output_path, intro_sec, outro_sec):
    try:
        duration = get_video_duration(input_path)
        print(f"视频: {input_path} 总时长: {duration:.2f} 秒")
        start = intro_sec
        end = duration - outro_sec
        if end <= start:
            raise ValueError("视频太短，无法剪裁")

        command = [
            "ffmpeg",
            "-ss", str(start),
            "-to", str(end),
            "-i", input_path,
            "-c", "copy",  # 直接拷贝不重新编码
            output_path
        ]
        
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[剪辑] {input_path} → {output_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"[错误] 剪辑失败: {input_path}")
        return False

def collect_videos(root_dir):
    grouped = {}
    for dirpath, _, filenames in os.walk(root_dir):
        mp4s = [os.path.join(dirpath, fn) for fn in filenames if fn.lower().endswith('.mp4')]
        if mp4s:
            grouped[dirpath] = sorted(mp4s)
    return grouped

def batch_trim_videos(input_dir, intro_sec=0, outro_sec=0):
    grouped = collect_videos(input_dir)
    for vdir in sorted(grouped.keys()):
        files = grouped[vdir]
        for idx, input_path in enumerate(files, start=1):
            output_path = os.path.join(vdir, f"{idx:02d}.mp4")
            if os.path.abspath(input_path) == os.path.abspath(output_path):
                temp_output = os.path.join(vdir, f"trim_{idx:02d}.mp4") # 临时文件
                success = trim_video(input_path, temp_output, intro_sec, outro_sec)
                if success:
                    os.remove(input_path)
                    os.replace(temp_output, output_path)
            else:
                success = trim_video(input_path, output_path, intro_sec, outro_sec)
                if success:
                    os.remove(input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查找所有根目录下的视频并裁剪片头片尾")
    parser.add_argument("--root", type=str, default="/nfs/yanzhang.ljx/workspace/datasets/YingShi/raw_zh", help="视频所在的根目录")
    parser.add_argument("--intro_sec", type=int, default=300, help="片头裁剪时长")
    parser.add_argument("--outro_sec", type=int, default=300, help="片尾裁剪时间")
    args = parser.parse_args()

    batch_trim_videos(args.root, intro_sec=300, outro_sec=300)
    print("✅ All videos trim processed.")

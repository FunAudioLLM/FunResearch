import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_list", required=True)
    parser.add_argument("--video_list", required=True)
    parser.add_argument("--overlap_list", required=True)
    parser.add_argument("--clean_wav_list", required=True)
    parser.add_argument("--clean_video_list", required=True)
    args = parser.parse_args()

    # 读取 overlap.list，提取文件名(不含扩展名)
    with open(args.overlap_list, "r") as f:
        overlap_names = {
            os.path.splitext(os.path.basename(line.strip()))[0]
            for line in f if line.strip()
        }

    # 从 wav.list 过滤
    with open(args.wav_list, "r") as f:
        wav_files = [line.strip() for line in f if line.strip()]
    clean_wav_files = [
        p for p in wav_files
        if os.path.splitext(os.path.basename(p))[0] not in overlap_names
    ]

    # 从 video.list 过滤
    with open(args.video_list, "r") as f:
        video_files = [line.strip() for line in f if line.strip()]
    clean_video_files = [
        p for p in video_files
        if os.path.splitext(os.path.basename(p))[0] not in overlap_names
    ]

    # 写入输出
    with open(args.clean_wav_list, "w") as f:
        for p in clean_wav_files:
            f.write(p + "\n")
    with open(args.clean_video_list, "w") as f:
        for p in clean_video_files:
            f.write(p + "\n")

if __name__ == "__main__":
    main()

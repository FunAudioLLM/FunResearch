# """
# This script uses pretrained models to perform speaker visual analysis.
# This script use following open source models:
#     1. Face detection: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
#     2. Active speaker detection: TalkNet, https://github.com/TaoRuijie/TalkNet-ASD
#     3. Face quality assessment: https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa
#     4. Face recognition: https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
#     5. Lip extraction: https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip
# """    

###################### cpu多线程版本 ##########################
import os
import json
import argparse
from glob import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from vision_processer import VisionProcesser, SharedModels
from speakerlab.utils.config import yaml_config_loader

parser = argparse.ArgumentParser(description='Extract visual speaker embeddings for diarization (no SharedModels).')
parser.add_argument('--conf', required=True, help='Config file')
parser.add_argument('--videos', required=True, help='Root dir to search for clean_video.list recursively')
parser.add_argument('--onnx_dir', default='', type=str, help='Pretrained onnx directory')
parser.add_argument('--debug_dir', default='', type=str, help='Debug video output directory')
parser.add_argument('--workers', type=int, default=32, help='Number of worker threads')
args = parser.parse_args()


def find_video_lists(root_dir):
    return glob(os.path.join(root_dir, '**', 'clean_video.list'), recursive=True)


def load_video_lists(root):
    lists = find_video_lists(root)
    all_videos = []
    video_metadata = {}
    for video_list in lists:
        episode_dir = os.path.dirname(video_list)
        film_name = os.path.basename(os.path.dirname(episode_dir))
        vad_path = os.path.join(episode_dir, 'json', 'vad.json')
        output_dir = os.path.join(episode_dir, 'embs_video')
        with open(video_list, 'r') as f:
            episode_videos = [line.strip() for line in f.readlines()]
        for vpath in episode_videos:
            if not os.path.isabs(vpath):
                raise ValueError(f"Video path {vpath} is not an absolute path: {vpath}")
            all_videos.append(vpath)
            video_metadata[vpath] = {'film_name': film_name, 'vad_path': vad_path, 'output_dir': output_dir}
    return all_videos, video_metadata


def merge_overlap_region(vad_time_list):
    if not vad_time_list:
        return []
    vad_time_list.sort(key=lambda x: x[0])
    out_vad_time_list = []
    for t in vad_time_list:
        if len(out_vad_time_list) == 0 or t[0] > out_vad_time_list[-1][1]:
            out_vad_time_list.append(t[:])
        else:
            out_vad_time_list[-1][1] = max(out_vad_time_list[-1][1], t[1])
    return out_vad_time_list


def create_debug_path(debug_dir, vpath):
    if not debug_dir:
        return None
    video_name = os.path.splitext(os.path.basename(vpath))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = os.path.join(debug_dir, f"{video_name}_DEBUG_{timestamp}.mp4")
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    return debug_path


def worker_process(vpath, metadata, shared_models, conf, args):
    film_name = metadata['film_name']
    vad_path = metadata['vad_path']
    output_dir = metadata['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    rec_id = os.path.splitext(os.path.basename(vpath))[0]
    embs_out_path = os.path.join(output_dir, f'{rec_id}.pkl')

    if os.path.isfile(embs_out_path):
        print(f"[INFO] Embeddings already exist: {embs_out_path}. Skipping.")
        return None

    if not os.path.isfile(vad_path):
        print(f"[WARNING] VAD file not found: {vad_path}. Skipping.")
        return None

    with open(vad_path, 'r') as f:
        vad_json = json.load(f)

    subset = {}
    for key in vad_json:
        k = str(key)
        if k.rsplit('_', 2)[0] == rec_id:
            subset[key] = vad_json[key]
    if len(subset) == 0:
        print(f"[WARNING] No VAD segments for {rec_id} in {vad_path}. Skipping.")
        return None

    rec_vad_time_list = [[v['start'], v['stop']] for v in subset.values()]
    rec_vad_time_list = merge_overlap_region(rec_vad_time_list)

    audio_path = os.path.splitext(vpath)[0] + '.wav'
    if not os.path.isfile(audio_path):
        print(f"[WARNING] Audio file missing: {audio_path}. Skipping.")
        return None

    debug_video_path = create_debug_path(args.debug_dir, vpath)
    if debug_video_path:
        print(f"[DEBUG] Debug video will be saved to: {debug_video_path}")

    vproc = VisionProcesser(
        video_file_path=vpath,
        audio_file_path=audio_path,
        audio_vad=rec_vad_time_list,
        out_feat_path=embs_out_path,
        shared_models=shared_models,
        conf=conf,
        out_video_path=debug_video_path
    )
    try:
        vproc.run()
        return embs_out_path
    except Exception as e:
        print(f"[ERROR] Failed processing {vpath}: {e}")
        raise
    finally:
        vproc.close()
        del vproc


def main():
    conf = yaml_config_loader(args.conf)
    all_videos, video_metadata = load_video_lists(args.videos)
    print(f"Found {len(all_videos)} videos.")
    if len(all_videos) == 0:
        return
    
    pool_sizes = {
        'face': 5,
        'asd' : 5,
        'fr'  : 30,
        'fq'  : 1,
        'lip' : 80,
    }
    
    shared_models = SharedModels(
        onnx_dir=args.onnx_dir, 
        device='cpu', 
        device_id=0,
        pool_sizes=pool_sizes
    )

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        for vpath in all_videos:
            meta = video_metadata[vpath]
            fut = exe.submit(worker_process, vpath, meta, shared_models, conf, args)
            futures[fut] = vpath

        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
            except Exception as e:
                print(f"[FAILED] {v} : {e}")
    print("All videos processed successfully!")


if __name__ == '__main__':
    main()


###################### gpu多卡版本 ##########################

# import os
# import json
# import argparse
# from glob import glob
# import torch
# import torch.distributed as dist
# from datetime import datetime

# from vision_processer import VisionProcesser
# from speakerlab.utils.config import yaml_config_loader

# parser = argparse.ArgumentParser(description='Extract visual speaker embeddings for diarization.')
# parser.add_argument('--conf', default=None, help='Config file')
# parser.add_argument('--videos', default=None, help='Video list file')
# parser.add_argument('--onnx_dir', default='', type=str, help='Pretrained onnx directory')
# parser.add_argument('--debug_dir', default='', type=str, help='Debug video output directory')
# parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')

# def merge_overlap_region(vad_time_list):
#     vad_time_list.sort(key=lambda x: x[0])
#     out_vad_time_list = []
#     for time in vad_time_list:
#         if len(out_vad_time_list)==0 or time[0] > out_vad_time_list[-1][1]:
#             out_vad_time_list.append(time)
#         else:
#             out_vad_time_list[-1][1] = time[1]
#     return out_vad_time_list

# def find_video_lists(root_dir):
#     """递归查找所有 clean_video.list文件"""
#     return glob(os.path.join(root_dir, '**', 'clean_video.list'), recursive=True)

# def create_debug_path(debug_dir, vpath):
#     """创建调试视频输出路径"""
#     if not debug_dir:
#         return None
#     video_name = os.path.splitext(os.path.basename(vpath))[0]
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     debug_path = os.path.join(debug_dir, f"{video_name}_DEBUG_{timestamp}.mp4")
#     os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    
#     return debug_path


# def main():
#     args = parser.parse_args()
#     conf = yaml_config_loader(args.conf)
#     dist.init_process_group(backend='gloo')
#     local_rank = int(os.environ['LOCAL_RANK'])
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
#     print(f"Rank {rank}/{world_size} started,  using GPU {local_rank}")

#     if args.use_gpu and torch.cuda.is_available():
#         if local_rank < torch.cuda.device_count():
#             torch.cuda.set_device(local_rank)
#             device = 'cuda'
#         else:
#             print(f"[WARNING] Local rank {local_rank} exceeds available GPUs. Using CPU.")
#             device = 'cpu'
#     else:
#         device = 'cpu'
    
#     video_lists = find_video_lists(args.videos)
#     if not video_lists:
#         if rank == 0:
#             raise Exception("No video.list files found in the directory: " + args.videos)
#         dist.barrier()
#         dist.destroy_process_group()
#         sys.exit(1)
        
#     # 收集所有视频文件路径及其对应的元数据
#     if rank == 0:
#         all_videos = []
#         video_metadata = {}  # 存储每个视频的vad路径和输出目录
#         for video_list in video_lists:
#             # 获取剧集目录（video.list所在目录）
#             episode_dir = os.path.dirname(video_list)
#             # 构建VAD文件路径（在json子目录下）
#             vad_path = os.path.join(episode_dir, 'json', 'vad.json')
#             # 构建输出目录（在embs_video子目录下）
#             output_dir = os.path.join(episode_dir, 'embs_video')
            
#             # 读取video.list中的视频文件
#             with open(video_list, 'r') as f:
#                 episode_videos = [line.strip() for line in f.readlines()]
            
#             for vpath in episode_videos:
#                 # 确保视频路径是绝对路径
#                 if not os.path.isabs(vpath):
#                     raise ValueError(f"Video path {vpath} is not an absolute path.")
#                 all_videos.append(vpath)
#                 video_metadata[vpath] = {
#                     'vad_path': vad_path,
#                     'output_dir': output_dir
#                 }
#         data = [all_videos, video_metadata]
#     else:
#         data = [None, None]
                
#     # 广播数据到所有进程
#     dist.broadcast_object_list(data, src=0)
#     all_videos, video_metadata = data[0], data[1]
            
#     # 分配任务给当前进程
#     local_videos = all_videos[rank::world_size]
#     print(f"Rank {rank} processing {len(local_videos)} videos")
#     # 处理每个分配的视频
#     for vpath in local_videos:
#         metadata = video_metadata[vpath]
#         vad_path = metadata['vad_path']
#         output_dir = metadata['output_dir']
        
#         # 创建输出目录
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 获取视频ID（不带扩展名的文件名）
#         rec_id = os.path.splitext(os.path.basename(vpath))[0]
#         embs_out_path = os.path.join(output_dir, f'{rec_id}.pkl')
        
#         # 如果已处理则跳过
#         # if os.path.isfile(embs_out_path):
#         #     print(f"[INFO] Embeddings already exist: {embs_out_path}. Skipping.")
#         #     continue
        
#         # 加载VAD数据
#         if not os.path.isfile(vad_path):
#             print(f"[WARNING] VAD file not found: {vad_path}. Skipping.")
#             continue
        
#         with open(vad_path, 'r') as f:
#             vad_json = json.load(f)
#         # 提取当前视频的VAD片段
#         subset = {}
#         for key in vad_json:
#             k = str(key)
#             if k.rsplit('_', 2)[0]==rec_id:
#                 subset[key] = vad_json[key]
#         if len(subset) == 0:
#             print(f"[WARNING] No VAD segments found for {rec_id} in {vad_path}. Skipping.")
#             continue
        
#         rec_vad_time_list = [[v['start'], v['stop']] for v in subset.values()]
#         rec_vad_time_list = merge_overlap_region(rec_vad_time_list)
#         # 构建音频路径（同目录同名.wav文件）
#         audio_path = os.path.splitext(vpath)[0] + '.wav'
#         # 创建debug视频输出路径
#         debug_video_path = create_debug_path(args.debug_dir, vpath)
#         if debug_video_path:
#             print(f"[DEBUG] Debug video will be saved to: {debug_video_path}")
            
#         # 处理视频
#         vprocesser = VisionProcesser(
#             vpath, 
#             audio_path, 
#             rec_vad_time_list, 
#             embs_out_path, 
#             args.onnx_dir, 
#             conf, 
#             device, 
#             local_rank,
#             debug_video_path
#         )
#         vprocesser.run()
#     # 等待所有进程完成
#     dist.barrier()
#     if rank == 0:
#         print("All videos processed successfully!")
#     # 清理分布式环境
#     dist.destroy_process_group()

# if __name__ == '__main__':
#     main()
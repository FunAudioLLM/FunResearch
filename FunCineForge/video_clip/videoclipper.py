import os
import sys
import librosa
import logging
import argparse
from moviepy.editor import VideoFileClip
import concurrent.futures
from utils.subtitle_utils import generate_srt, generate_srt_clip
from utils.argparse_tools import ArgumentParser
from utils.trans_utils import write_state, load_state, convert_pcm_to_float
from funasr import AutoModel
# If you find that the generated srt file is too fragmented, you need to add "if punc_id > 2:" to line 167 after "sentence_text += punc_list[punc_id - 2]" in funasr 1.2.7 funasr.utils.timestamp_tools file.

class VideoClipper():
    def __init__(self, funasr_model):
        logging.info("Initializing VideoClipper.")
        self.funasr_model = funasr_model
        self.GLOBAL_COUNT = 0
        self.lang = 'zh'

    def recog(self, audio_input, sd_switch='no', state=None, hotwords="", output_dir=None):
        if state is None:
            state = {}
        sr, data = audio_input

        # Convert to float64 consistently (includes data type checking)
        data = convert_pcm_to_float(data)

        # assert sr == 16000, "16kHz sample rate required, {} given.".format(sr)
        if sr != 16000: # resample with librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        if len(data.shape) == 2:  # multi-channel wav input
            logging.warning("Input wav shape: {}, mean channels.".format(data.shape))
            data = data.mean(axis=1)
        state['audio_input'] = (sr, data)
        if sd_switch == 'yes':
            rec_result = self.funasr_model.generate(data, 
                                                    return_spk_res=True,
                                                    return_raw_text=True, 
                                                    is_final=True,
                                                    output_dir=output_dir, 
                                                    hotword=hotwords, 
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={},
                                                    merge_vad=True,               # 开启 VAD 合并
                                                    merge_length_s=30             # 设置合并目标段最大长度（秒）
                                                    )
            res_srt = generate_srt(rec_result[0]['sentence_info'])
        else:
            rec_result = self.funasr_model.generate(data, 
                                                    return_spk_res=False, 
                                                    sentence_timestamp=True, 
                                                    return_raw_text=True, 
                                                    is_final=True, 
                                                    hotword=hotwords,
                                                    output_dir=output_dir,
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={},
                                                    merge_vad=True,               # 开启 VAD 合并
                                                    merge_length_s=30             # 设置合并目标段最大长度（秒）
                                                    )
            res_srt = generate_srt(rec_result[0]['sentence_info'])
        state['recog_res_raw'] = rec_result[0]['raw_text']
        state['timestamp'] = rec_result[0]['timestamp']
        state['sentences'] = rec_result[0]['sentence_info']
        res_text = rec_result[0]['text']
        del data  # clear memory
        return res_text, res_srt, state
    

    def video_recog(self, video_filename, sd_switch='no', hotwords="", output_dir=None):
        video = VideoFileClip(video_filename)
        # Extract the base name, add '_clip.mp4', and 'wav'
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
            audio_file = base_name + '.wav'
            audio_file = os.path.join(output_dir, audio_file)
        else:
            base_name, _ = os.path.splitext(video_filename)
            audio_file = base_name + '.wav'

        if video.audio is None:
            logging.error("No audio information found.")
            sys.exit(1)
        
        video.audio.write_audiofile(audio_file, verbose=False, logger=None)
        wav = librosa.load(audio_file, sr=16000)[0]
        if os.path.exists(audio_file):
            os.remove(audio_file)
        state = {
            'video_filename': video_filename
        }
        video.close()
        del video
        return self.recog((16000, wav), sd_switch, state, hotwords, output_dir)

    def video_clip(self, state, output_dir=None):
        """
        Clip the video based on the given dest_text or provided timestamps in the state.
        """
        # Retrieve data from the state
        sentences = state['sentences']
        video_filename = state['video_filename']
        
        # timestamps
        ts = []
        for sentence in sentences:
            start_time = sentence['start'] / 1000.0  # Convert to seconds
            end_time = sentence['end'] / 1000.0  # Convert to seconds
            speaker_id = sentence.get('spk', 'unknown')  # Get speaker id from the sentence
            ts.append([start_time, end_time, speaker_id])
        srt_index = 1
        time_acc_ost = 0.0

        # If there are timestamps, proceed with video clipping
        if len(ts):
            time_acc_ost = 0.0
            for i, (start, end, speaker_id) in enumerate(ts):
                clipped_folder = os.path.join(output_dir, 'clipped')
                os.makedirs(clipped_folder, exist_ok=True)

                # Create filename
                srt_clip, subs, srt_index = generate_srt_clip(
                    sentences, start, end, begin_index=srt_index-1, time_acc_ost=time_acc_ost
                )
                base_name = os.path.basename(video_filename)
                video_name_without_ext, _ = os.path.splitext(base_name)
                start_hours = int(subs[0][0][0] // 3600)
                start_minutes = int((subs[0][0][0] % 3600) // 60)
                start_seconds = int(subs[0][0][0] % 60)
                start_milliseconds = int((subs[0][0][0] - int(subs[0][0][0])) * 100)  # Extract milliseconds
                clip_filename = f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_{start_milliseconds:02}_spk{speaker_id}"
                clip_filepath = os.path.join(clipped_folder, clip_filename)
        
                # Clip the video and Write the video clip
                video_filepath = clip_filepath + '.mp4'
                audio_filepath = clip_filepath + '.wav'
                clip_srt_file = clip_filepath + '.srt'
                if not (os.path.exists(video_filepath) and os.path.exists(audio_filepath) and os.path.exists(clip_srt_file)):
                    print(f"[CLIP] {video_filepath}.")
                    with VideoFileClip(video_filename) as video:
                        sub = video.subclip(start, end)
                        sub.write_videofile(video_filepath, audio_codec="aac", verbose=False, logger=None)
                        sub.audio.write_audiofile(audio_filepath, codec='pcm_s16le', verbose=False, logger=None)
                        sub.close()
                        del sub
                    # Write the SRT file
                    with open(clip_srt_file, 'w') as fout:
                        fout.write(srt_clip) 

                time_acc_ost += (end - start)
            
            message = f"{len(ts)} periods found in the speech, clips created."
        else:
            message = "[WARNING] No valid periods found in the speech."

        return message


def runner(stage, file, sd_switch, output_dir, audio_clipper):
    audio_suffixs = ['.wav','.mp3','.aac','.m4a','.flac']
    video_suffixs = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    _,ext = os.path.splitext(file)
    if ext.lower() in audio_suffixs:
        mode = 'audio'
    elif ext.lower() in video_suffixs:
        mode = 'video'
    else:
        logging.error("Unsupported file format: {}\n\nplease choise one of the following: {}".format(file),audio_suffixs+video_suffixs)
        sys.exit(1) # exit if the file is not supported
    while output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    if stage == 1:
        if mode == 'audio':
            wav, sr = librosa.load(file, sr=16000)
            res_text, res_srt, state = audio_clipper.recog((sr, wav), sd_switch)
        if mode == 'video':
            res_text, res_srt, state = audio_clipper.video_recog(file, sd_switch)
        total_srt_file = output_dir + '/total.srt'
        with open(total_srt_file, 'w') as fout:
            fout.write(res_srt)
        write_state(output_dir, state)
        print("Recognition successed. You can copy the text segment from below and use stage 2.")
        
    if stage == 2:
        if mode == 'video':
            state = load_state(output_dir)
            state['video_filename'] = file
            message = audio_clipper.video_clip(state, output_dir=output_dir)
            print("Clipping Log: {}".format(message))
            
       
def find_all_videos(folder, base_output_dir=None, skip_processed=True, suffixes=None):
    if suffixes is None:
        suffixes = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    all_videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in suffixes):
                file_path = os.path.join(root, file)
                
                # 检查是否需要跳过已处理的文件
                if skip_processed and base_output_dir is not None:
                    # 构建预期的输出目录结构
                    parent_dir_name = os.path.basename(root)
                    video_name = os.path.splitext(file)[0]
                    output_subdir = os.path.join(base_output_dir, parent_dir_name, video_name)
                    
                    # 检查处理完成的标记文件
                    total_srt = os.path.join(output_subdir, 'total.srt')
                    
                    # 如果标记文件存在，则跳过已处理的文件
                    if os.path.exists(total_srt):
                        print(f"Skipping already processed file: {file_path}")
                        continue
                
                all_videos.append(file_path)
    return all_videos


def process_single_video(file, stage, sd_switch, base_output_dir, audio_clipper=None):
    try:
        video_name = os.path.splitext(os.path.basename(file))[0]
        parent_dir_name = os.path.basename(os.path.dirname(file))
        output_dir = os.path.join(base_output_dir, parent_dir_name, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        runner(stage=stage, file=file, sd_switch=sd_switch, output_dir=output_dir, audio_clipper=audio_clipper)
        print(f"✅ Done: {file}")
    except Exception as e:
        logging.error(f"❌ Failed: {file}, error: {e}")
        
def init_models(lang='zh'):
    if lang == 'zh':
        funasr_model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common"
        )
    elif lang == 'en':
        funasr_model = AutoModel(
            model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common"
        )
    return funasr_model

def get_parser():
    parser = ArgumentParser(
        description="ClipVideo Argument",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        help="Stage, 0 for recognizing and 1 for clipping",
        required=True
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file or folder",
        required=True
    )
    parser.add_argument(
        "--sd_switch",
        type=str,
        choices=("no", "yes"),
        default="no",
        help="Turn on the speaker diarization or not",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
        help="Output files path",
    )
    parser.add_argument(
        "--skip_processed",
        action="store_true",
        help="If set, skip processed for stage 1",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default='zh',
        help="language"
    )
    return parser

def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    
    file_or_folder = kwargs['file']
    stage = kwargs['stage']
    sd_switch = kwargs['sd_switch']
    output_dir = kwargs['output_dir']
    lang = kwargs['lang']
    
    # Initialize models
    if stage == 1:
        funasr_model = init_models(lang)
    else:
        funasr_model = None
    audio_clipper = VideoClipper(funasr_model)
    audio_clipper.lang = lang
    
    if os.path.isdir(file_or_folder):
        all_videos = find_all_videos(file_or_folder, base_output_dir=output_dir, skip_processed=kwargs['skip_processed'])
        print(f"Found {len(all_videos)} video files.")

        # 多线程并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(process_single_video, file, stage, sd_switch, output_dir, audio_clipper)
                       for file in all_videos]
            concurrent.futures.wait(futures)
        print("✅ All videos processed.")
    else:
        # 单个文件处理
        runner(stage=stage, file=file_or_folder, sd_switch=sd_switch, output_dir=output_dir, audio_clipper=audio_clipper)


if __name__ == '__main__':
    main()

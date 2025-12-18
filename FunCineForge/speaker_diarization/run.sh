set -e
. ./path.sh || exit 1

stage=1
stop_stage=4
hf_access_token=hf_xxx
root=/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh
gpus="-1"
conf_file=conf/diar_video.yaml
audio_conf_file=conf/diar.yaml
pretrained_models=pretrained_models

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)        stage="$2"     ;;
        --stop_stage)   stop_stage="$2";;
        --hf_access_token) hf_access_token="$2" ;;
        --root)         root="$2"      ;;
        --gpus)         gpus="$2"      ;;
        *)
            echo "未知参数: $1" >&2
            exit 1
            ;;
    esac
    shift 2 || { echo "缺少参数值: $1" >&2; exit 1; }
done

echo stage $stage to $stop_stage, root $root, gpus $gpus
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "$(basename $0) Stage1: Make list..."
  find "$root" -type d -name "clipped" | while read clipped_dir; do
    # 上一级目录
    parent_dir=$(dirname "$clipped_dir")
    # 写入 mp4 路径到 video.list
    find "$clipped_dir" -type f -name "*.mp4" | sort > "${parent_dir}/video.list"
    # 写入 wav 路径到 wav.list
    find "$clipped_dir" -type f -name "*.wav" | sort > "${parent_dir}/wav.list"
  done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  audio_stage=1
  audio_stop_stage=5
  echo "$(basename $0) Stage2: Extract audio speaker embeddings..."
  find "$root" -type f -name "wav.list" | while read wav_list; do
    dir=$(dirname "$wav_list")
    json_dir=$dir/json
    embs_dir=$dir/embs_wav

    if [ ${audio_stage} -le 1 ] && [ ${audio_stop_stage} -ge 1 ]; then
      if [ ! -f "$dir/overlap.list" ]; then
        echo "$(basename $0) Audio Stage1: Do overlap detection for $wav_list ..."
        if [ "$gpus" = "-1" ]; then
          python local/overlap_detection.py \
              --wavs $wav_list \
              --out_dir $dir \
              --hf_access_token $hf_access_token \
              --overlap_threshold 0.8
        else
          nj=$(echo $gpus | wc -w)
          python local/overlap_detection.py \
              --wavs $wav_list \
              --out_dir $dir \
              --hf_access_token $hf_access_token \
              --overlap_threshold 0.8 \
              --nj $nj \
              --use_gpu
        fi
      fi
    fi

    if [ ${audio_stage} -le 2 ] && [ ${audio_stop_stage} -ge 2 ]; then
      if [ ! -f "${dir}/clean_wav.list" ] || [ ! -f "${dir}/clean_video.list" ]; then
        echo "$(basename $0) Audio Stage2: Generate clean_wav.list and clean_video.list ..."
        python local/filter_clean_list.py \
            --wav_list "$wav_list" \
            --video_list "${dir}/video.list" \
            --overlap_list "${dir}/overlap.list" \
            --clean_wav_list "${dir}/clean_wav.list" \
            --clean_video_list "${dir}/clean_video.list"
      fi
    fi

    if [ ${audio_stage} -le 3 ] && [ ${audio_stop_stage} -ge 3 ]; then
      if [ ! -f "$json_dir/vad.json" ]; then
        mkdir -p $json_dir
        echo "$(basename $0) Audio Stage3: Do vad for clean_wav.list ..."
        python local/voice_activity_detection.py --wavs "${dir}/clean_wav.list" --out_file $json_dir/vad.json
      fi
    fi

    if [ ${audio_stage} -le 4 ] && [ ${audio_stop_stage} -ge 4 ]; then
      if [ ! -f "$json_dir/subseg.json" ]; then
        mkdir -p $json_dir
        echo "$(basename $0) Audio Stage4: Prepare subsegments info for clean_wav.list ..."
        python local/prepare_subseg_json.py --vad $json_dir/vad.json --out_file $json_dir/subseg.json
      fi
    fi

    if [ ${audio_stage} -le 5 ] && [ ${audio_stop_stage} -ge 5 ]; then
      if [ ! -d "$embs_dir" ] || [ -z "$(ls -A "$embs_dir" 2>/dev/null)" ]; then
        mkdir -p $embs_dir
        echo "$(basename $0) Audio Stage5: Extract speaker embeddings for clean_wav.list ..."
        # Set speaker_model_id to iic/speech_campplus_sv_zh_en_16k-common_advanced if using snapshot_download
        # speaker_model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
        if [ "$gpus" = "-1" ]; then
          python local/extract_speech_embeddings.py \
                --pretrained_model $pretrained_models/speech_campplus --conf $audio_conf_file \
                --subseg_json $json_dir/subseg.json --embs_out $embs_dir --rank -1 --world_size 1
        else
          nj=$(echo $gpus | wc -w)
          torchrun --nproc_per_node=$nj local/extract_speech_embeddings.py \
                --pretrained_model $pretrained_models/speech_campplus --conf $audio_conf_file \
                --subseg_json $json_dir/subseg.json --embs_out $embs_dir --gpu $gpus --use_gpu
        fi
      fi
    fi
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "$(basename $0) Stage3: Extract visual speaker embeddings..."
  python local/extract_visual_embeddings.py --conf $conf_file --videos $root \
           --onnx_dir $pretrained_models --workers 64
           # --debug_dir /nfs/yanzhang.ljx/workspace/datasets/YingShi/debug_videos
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "$(basename $0) Stage4: Clustering for both type of speaker embeddings..."
  torchrun --nproc_per_node=50 local/cluster_and_postprocess.py --conf $conf_file --root $root
fi

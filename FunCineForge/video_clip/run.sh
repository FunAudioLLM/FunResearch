stage=1
stop_stage=2
input=/nfs/yanzhang.ljx/workspace/datasets/YingShi/raw_zh
output=/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh
lang='zh'

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)        stage="$2"     ;;
        --stop_stage)   stop_stage="$2";;
        --input)        input="$2" ;;
        --output)       output="$2"      ;;
        --lang)         lang="$2"      ;;
        *)
            echo "未知参数: $1" >&2
            exit 1
            ;;
    esac
    shift 2 || { echo "缺少参数值: $1" >&2; exit 1; }
done

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        echo "$(basename $0) Stage1: Generate srt and timestamp"
        python videoclipper.py \
                --stage 1 \
                --file $input \
                --output_dir $output \
                --sd_switch "yes" \
                --lang $lang \
                --skip_processed
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "$(basename $0) Stage2: Trim long videos by punctuation marks"
        python videoclipper.py \
                --stage 2 \
                --file $input \
                --output_dir $output \
                --lang $lang
fi
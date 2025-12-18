# Video Clip

## Introduction
This recipe offers methods for automatically clipping long videos. It uses the FunASR series of models to perform speech recognition in the video, then uses voice activity detection and merging to segment the speech into sentence-level segments of the target duration. A second stage of cropping then batches all the video segments.

## Usage
### Quick Start
```
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh
```
## Illustrate
- If you find that the generated srt file is too fragmented, you need to add "if punc_id > 2:" to line 167 after "sentence_text += punc_list[punc_id - 2]" in funasr 1.2.7 funasr.utils.timestamp_tools file.
- Model and data loading may take a long time, please be patient.
- The directory structure is as follows:
```
datasets/clean/zh (output dir)
│
└───film name 1
│   │
│   └───01
│   │   │   total.srt
│   │   │   sentences
│   │   │   timestamp
│   │   │   recog_res_raw
│   │   └───clipped
│   │       │   01_00_00_00_00_spk0.mp4
│   │       │   01_00_00_00_00_spk0.srt
│   │       │   01_00_00_00_00_spk0.wav
│   │
│   └───02
│   │   │   total.srt
│   │   │   sentences
│   │   │   timestamp
│   │   │   recog_res_raw
│   │   └───clipped
│   │       │   02_00_00_00_00_spk0.mp4
│   │       │   02_00_00_00_00_spk0.srt
│   │       │   02_00_00_00_00_spk0.wav
│   │   ...
│   
└───film name 2
│   │
│   └───01
│   ...
```

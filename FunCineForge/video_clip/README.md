# Video Clip

## Introduction
This recipe offers methods for automatically clipping long videos. It uses the FunASR series of models to perform speech recognition in the video, then uses voice activity detection and merging to segment the speech into sentence-level segments of the target duration. A second stage of cropping then batches all the video segments.

It supports long videos in both Chinese and English, multi-threaded CPU and multi-GPU operation, and multi-machine operation.

## Usage
### Quick Start
```
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh --device cpu
```
To run this script on multiple machines, you need to execute the following script on each machine.
```
bash run_dist.sh -stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh --device cpu --machine_rank 0 --total_machines 3
```

lang: **zh** or **en**

device: **cpu** or **gpu(cuda)**

machine_rank: **0** or **1** or **2** corresponds to **3** total_machines

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

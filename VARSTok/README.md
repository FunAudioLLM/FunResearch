# VARSTok
A fully dynamic, variable-frame-rate speech tokenizer that can be seamlessly integrated into LLMs. 

This paper has been accepted as an **oral** presentaion at **AAAI 2026**.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2509.04685)
[![demo](https://img.shields.io/badge/VARSTok-Demo-red)](https://zhengrachel.github.io/VARSTok/)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20VARSTok-Models-blue)](https://huggingface.co/ZhengRachel/VARSTok)

## ⚠️ Important Note on Clustering Implementation
This repository contains two versions of the clustering module in ```encoder/clustering_acc.py```:

```SequentialDPClustering```: This is the original version used to train the models for our paper. It contains a minor bug (an incorrect negative sign in the calculation of _local_density and _delta). To reproduce the paper's results with the provided checkpoints, please use this default version. Simply run the code as instructed. Our provided weights match this implementation.

```SequentialDPClusteringFixed```: This is the corrected version of the algorithm. If you plan to retrain the model or for future research, we strongly recommend using this fixed version. You can activate it by uncommenting [line 100 in ```decoder/feature_extractors.py```](decoder/feature_extractors.py#L100). Directly using the provided weights with this fixed version for inference will lead to a slight but not significant performance drop.

## 🚀 Installation

1. Create and activate a new Conda environment:
    ```bash
    conda create -n varstok python=3.9
    conda activate varstok
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Usage Examples (Inference)

### Part1: Reconstruct speech from a raw WAV File

This example shows the full encode-decode loop.

```python

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import VARSTok


device=torch.device('cpu')

config_path = "./configs/xxx.yaml"
model_path = "./xxx.ckpt"
audio_outpath = "xxx"

varstok = VARSTok.from_pretrained(config_path, model_path)
varstok = varstok.to(device)


wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0]).to(device)
wav = wav.to(device)
features, discrete_code, cluster_lengths = varstok.encode_infer(wav, bandwidth_id=bandwidth_id)
audio_out = varstok.decode(features, cluster_lengths, bandwidth_id=bandwidth_id) 
torchaudio.save(audio_outpath, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
```

Alternatively, you can run the provided inference script:
```bash
python infer.py
```


### Part2: Generating discrete codes from speech
This shows how to get the token sequence (LLM input) from an audio file.
```python

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import VARSTok

device=torch.device('cpu')

config_path = "./configs/xxx.yaml"
model_path = "./xxx.ckpt"

varstok = VARSTok.from_pretrained(config_path, model_path)
varstok = varstok.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0]).to(device)
wav = wav.to(device)
_, discrete_code, _ = varstok.encode_infer(wav, bandwidth_id=bandwidth_id)
print(discrete_code)
```



### Part3: Speech reconstruction from discrete codes
```python
# audio_tokens
features, cluster_lengths = varstok.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([0]).to(device)  
audio_out = varstok.decode(features, cluster_lengths, bandwidth_id=bandwidth_id)
```

## 🏋️ Training

### Step1: Prepare train dataset
Please follow the data processing pipeline from the [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) repository (see Acknowledgements).

### Step2: Modifying configuration files
1. Open the config file ```./configs/xxx.yaml```.
2. Modify the parameters, especially
    - ```batch_size```
    - ```filelist_path```
    - ```save_dir```
    - ```device```
    
    The ```resume_config``` and ```resume_model``` parameters are for the pretrained 75 Hz WavTokenizer and can be obtained from [WavTokenizer's model page](https://huggingface.co/novateur/WavTokenizer).

### Step3: Start training process
Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

```bash
cd ./VARSTok
bash run.sh
```

You can control the degree of dynamic compression by adjusting the hyperparameters `threshold` and `max_span` in `encoder/clustering_acc.py`.

## 🙏 Acknowledgement
The codebase is heavily adapted from [WavTokenizer](https://github.com/jishengpeng/WavTokenizer). Thanks for their wonderful work.
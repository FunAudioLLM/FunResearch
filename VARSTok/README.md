# VARSTok
A fully dynamic, variable-frame-rate speech tokenizer that can be seamlessly integrated into LLMs.


## Installation

To use VARSTok, install it using:

```bash
conda create -n varstok python=3.9
conda activate varstok
pip install -r requirements.txt
```

## Infer

### Part1: Reconstruct audio from raw wav

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


### Part2: Generating discrete codecs
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



### Part3: Audio reconstruction through codecs
```python
# audio_tokens
features, cluster_lengths = varstok.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([0]).to(device)  
audio_out = varstok.decode(features, cluster_lengths, bandwidth_id=bandwidth_id)
```

## Training

### Step1: Prepare train dataset
```python
# Follow the data processing pipeline of WavTokenizer
```

### Step2: Modifying configuration files
```python
# ./configs/xxx.yaml
# Modify the values of parameters such as batch_size, filelist_path, save_dir, device
```

### Step3: Start training process
Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

```bash
cd ./VARSTok
bash run.sh
```

You can control the degree of dynamic compression by adjusting the hyperparameters `threshold` and `max_span` in `encoder/clustering_acc.py`.

## Acknowledgement
The codebase of SimVQ is adapted from [WavTokenizer](https://github.com/jishengpeng/WavTokenizer). Thanks for their wonderful work.
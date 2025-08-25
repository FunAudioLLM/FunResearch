# --coding:utf-8--
import os

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import VARSTok

import time

import logging
from tqdm import tqdm

device1=torch.device('cuda:0')
# device2=torch.device('cpu')

input_path = "/path/to/test_list"
out_folder = '/path/to/output'
ll=""

tmptmp=out_folder+"/"+ll

os.system("rm -r %s"%(tmptmp))
os.system("mkdir -p %s"%(tmptmp))

# 自己数据模型加载
config_path = "/path/to/config.yaml"
model_path = "/path/to/ckpt"
varstok = VARSTok.from_pretrained(config_path, model_path)
varstok = varstok.to(device1)

with open(input_path,'r') as fin:
    x=fin.readlines()

x = [i.strip() for i in x]
x = x[:]

# x= ["/cpfs_speech3/zhengruichen.zrc/WavTokenizer-copy/gaussian_white_noise_24000Hz.wav"]

# 完成一些加速处理

features_all=[]
cluster_lengths_all = []
tokens = 0.0
org_tokens = 0.0

for i in tqdm(range(len(x))):

    wav, sr = torchaudio.load(x[i])
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
    wav = resampler(wav)
    # print("***:",x[i])
    # wav = convert_audio(wav, sr, 24000, 1)                             # (1,131040)
    bandwidth_id = torch.tensor([0])
    wav=wav.to(device1)
    # print(i)

    try:
        features, discrete_code, cluster_lengths = varstok.encode_infer(wav, bandwidth_id=bandwidth_id)
        features_all.append(features)
        cluster_lengths_all.append(cluster_lengths)
        tokens += features.size(-1)
        org_tokens += cluster_lengths.sum() 
        # print(x[i], cluster_lengths, discrete_code, features.size(-1), cluster_lengths.sum(), tokens/org_tokens*75)
    except Exception as e:
        print(f"Error while processing wav: {x[i], e}")  

print("Average Frame Rate:", (tokens/org_tokens*75).item(), "Hz")

for i in tqdm(range(len(features_all))):

    bandwidth_id = torch.tensor([0])

    bandwidth_id = bandwidth_id.to(device1) 

    audio_out = varstok.decode(features_all[i], cluster_lengths_all[i], bandwidth_id=bandwidth_id)   
    audio_path = out_folder + '/' + ll + '/' + x[i].split('/')[-1]
    os.makedirs(out_folder + '/' + ll, exist_ok=True)
    torchaudio.save(audio_path, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)






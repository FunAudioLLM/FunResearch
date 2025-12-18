# Speech Separation

## Introduction
A [Mel-Band-Roformer Vocal model](https://arxiv.org/abs/2310.01809). This model performs slightly better than the paper equivalent due to training with more data.

## Usage
### Quick Start
GPU version (recommended)
```
python run.py --root datasets/clean/zh --gpus 0 1 2 3
```
If you want to use CPU multithreading, you can use the following command, but it will run very slowly
```
python run.py --root datasets/clean/zh
```

Config description: inference parameters.

[num_overlap](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/41d04ae1c8ea89261b488e90953192efe650fa4f/configs/config_vocals_mel_band_roformer.yaml#L38) - Increasing this value can improve the quality of the outputs due to helping with artifacts created when putting the chunks back together. This will make inference times longer (you don't need to go higher than 8)

[chunk_size](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/41d04ae1c8ea89261b488e90953192efe650fa4f/configs/config_vocals_mel_band_roformer.yaml#L39) - The length of audio input into the model (default is 352800 which is 8 seconds, 352800 was also used to train the model)





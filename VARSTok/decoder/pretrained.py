import os
from typing import Tuple, Any, Union, Dict

import torch
import yaml
from huggingface_hub import hf_hub_download
from torch import nn
from decoder.feature_extractors import FeatureExtractor, EncodecFeatures
from decoder.heads import FourierHead
from decoder.models import Backbone
from decoder.duration_predictor import LengthRegulator, DurationPredictor

def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class VARSTok(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self, feature_extractor: FeatureExtractor, backbone: Backbone, head: FourierHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
        # self.duration_predictor = DurationPredictor()
        self.length_regulator = LengthRegulator()


    @classmethod
    def from_hparams(cls, config_path: str) -> "Vocos":
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config['model']['init_args']["feature_extractor"])
        backbone = instantiate_class(args=(), init=config['model']['init_args']["backbone"])
        head = instantiate_class(args=(), init=config['model']['init_args']["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model


    @classmethod
    def from_pretrained(self, config_path, model_path):
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        model = self.from_hparams(config_path)
        state_dict_raw = torch.load(model_path, map_location="cpu")['state_dict']
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if k.startswith('backbone.') or k.startswith('head.') or k.startswith('feature_extractor.'):
                state_dict[k] = v
        model.load_state_dict(state_dict)
        model.eval()
        return model



    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features, _, _ = self.feature_extractor(audio_input, **kwargs)  # 0818
        audio_output = self.decode(features, **kwargs)
        return audio_output


    # 0818
    @torch.inference_mode()
    def encode_infer(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        features, discrete_codes, _, cluster_lengths = self.feature_extractor.infer(audio_input, **kwargs)
        
        B, L = cluster_lengths.shape

        shifted_codes = torch.full_like(discrete_codes, -1) 

        for b in range(B):
            for l in range(L):
                shift = cluster_lengths[b, l]
                if shift > 0:
                    shifted_codes[0, b, l] = discrete_codes[0, b, l] + (shift - 1) * 4096
                elif shift == 0:
                    shifted_codes[0, b, l] = -1
                else:
                    raise ValueError("cluster_lengths should not contain negative values")
                
        return features, shifted_codes, cluster_lengths


    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, cluster_lengths: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        if cluster_lengths is not None:
            features_input, _ = self.length_regulator(features_input, cluster_lengths)
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    @torch.inference_mode()
    def codes_to_features(self, shifted_codes: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature extractor's
        codebook weights.

        Args:
            codes (Tensor): The input tensor. Expected shape is (K, L) or (K, B, L),
                            where K is the number of codebooks, B is the batch size and L is the sequence length.

        Returns:
            Tensor: Features of shape (B, C, L), where B is the batch size, C denotes the feature dimension,
                    and L is the sequence length.
        """
        assert isinstance(
            self.feature_extractor, EncodecFeatures
        ), "Feature extractor should be an instance of EncodecFeatures"
        
        # pdb.set_trace()
        B, L = shifted_codes.shape[1], shifted_codes.shape[2]
    
        # 创建初始化为零的 cluster_lengths
        cluster_lengths = torch.zeros((B, L), dtype=torch.int32)
        
        # 创建一个新的 tensor 用于存储恢复后的 codes
        codes = torch.full_like(shifted_codes, -1)  # 初始化为 -1

        for b in range(B):
            for l in range(L):
                shifted_value = shifted_codes[0, b, l]
                if shifted_value == -1:
                    # 这个位置是无效的，cluster_lengths 为 0，code 为 -1
                    cluster_lengths[b, l] = 0
                    codes[0, b, l] = -1
                else:
                    # 计算 cluster_lengths 和恢复 codes
                    cluster_length = shifted_value // 4096 + 1
                    cluster_lengths[b, l] = cluster_length
                    original_code = shifted_value % 4096
                    codes[0, b, l] = original_code
        
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        n_bins = self.feature_extractor.encodec.quantizer.bins
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)

        tmp=torch.cat([vq.codebook for vq in self.feature_extractor.encodec.quantizer.vq.layers],dim=0)
        # features = torch.nn.functional.embedding(embeddings_idxs, self.feature_extractor.codebook_weights).sum(dim=0)
        features = torch.nn.functional.embedding(embeddings_idxs, tmp).sum(dim=0)
        features = features.transpose(1, 2)
        cluster_lengths = cluster_lengths.to(features.device)

        return features, cluster_lengths



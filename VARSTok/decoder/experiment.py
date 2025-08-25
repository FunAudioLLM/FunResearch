import math

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import transformers
import yaml

import torch.nn.functional as F

from decoder.discriminator_dac import DACDiscriminator

from decoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from decoder.feature_extractors import FeatureExtractor
from decoder.heads import FourierHead
from decoder.helpers import plot_spectrogram_to_numpy
from decoder.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss, DACGANLoss
from decoder.models import Backbone
from decoder.modules import safe_log
from decoder.duration_predictor import LengthRegulator, DurationPredictor, DurationScheduler

import pdb

class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        resume_config: str,
        resume_model: str,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        resume: bool = False,
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

        self.resume_config = resume_config
        self.resume_model = resume_model
        self.resume = resume

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        
        self.dac = DACDiscriminator()

        self.dacdiscriminator = DACGANLoss(self.dac)

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

        self.length_regulator = LengthRegulator()
        # self.duration_predictor = DurationPredictor()
        # self.duration_scheduler = DurationScheduler(start_step=100000, end_step=50000, max_step=2000000)
        # for param in self.duration_predictor.parameters():
        #     param.requires_grad = True


    # def configure_optimizers(self):
    #     disc_params = [
    #         {"params": self.multiperioddisc.parameters()},
    #         {"params": self.multiresddisc.parameters()},
    #         {"params": self.dac.parameters()},
    #     ]
    #     gen_params = [
    #         {"params": self.feature_extractor.parameters()},
    #         {"params": self.backbone.parameters()},
    #         {"params": self.head.parameters()},
    #     ]

    #     opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate)
    #     opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate)

    #     max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
    #     scheduler_disc = transformers.get_cosine_schedule_with_warmup(
    #         opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
    #     )
    #     scheduler_gen = transformers.get_cosine_schedule_with_warmup(
    #         opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
    #     )

    #     return (
    #         [opt_disc, opt_gen],
    #         [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
    #     )
    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters(), "lr": self.hparams.initial_learning_rate},
            {"params": self.multiresddisc.parameters(), "lr": self.hparams.initial_learning_rate},
            {"params": self.dac.parameters(), "lr": self.hparams.initial_learning_rate},
        ]

        gen_params = [
            {"params": self.feature_extractor.encodec.quantizer.parameters(), "lr": self.hparams.initial_learning_rate},  # VQ部分
            {"params": self.feature_extractor.encodec.encoder.parameters(), "lr": self.hparams.initial_learning_rate},  # encoder
            {"params": self.feature_extractor.encodec.decoder.parameters(), "lr": self.hparams.initial_learning_rate},  # encodec decoder (?)
            # {"params": self.feature_extractor.simvq_quantizer.parameters(), "lr": 2*self.hparams.initial_learning_rate},  # encodec decoder (?)
            # {"params": self.duration_predictor.parameters(), "lr": self.hparams.initial_learning_rate},                  # Duration predictor
            {"params": self.backbone.parameters(), "lr": self.hparams.initial_learning_rate},                              # Decoder (backbone)
            {"params": self.head.parameters(), "lr": self.hparams.initial_learning_rate},                                  # Decoder (head)
        ]

        opt_disc = torch.optim.AdamW(disc_params)
        opt_gen = torch.optim.AdamW(gen_params)

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [
                {"scheduler": scheduler_disc, "interval": "step"},
                {"scheduler": scheduler_gen, "interval": "step"},
            ],
        )
    # def configure_optimizers(self):
    #     disc_params = [
    #         {"params": self.multiperioddisc.parameters(), "lr": self.hparams.initial_learning_rate},
    #         {"params": self.multiresddisc.parameters(), "lr": self.hparams.initial_learning_rate},
    #         {"params": self.dac.parameters(), "lr": self.hparams.initial_learning_rate},
    #     ]

    #     gen_params = [
    #         {"params": self.feature_extractor.encodec.quantizer.parameters(), "lr": 2 * self.hparams.initial_learning_rate},  # VQ部分
    #         {"params": self.feature_extractor.encodec.encoder.parameters(), "lr": self.hparams.initial_learning_rate},  # encoder
    #         {"params": self.feature_extractor.encodec.decoder.parameters(), "lr": self.hparams.initial_learning_rate},  # encodec decoder
    #         {"params": self.duration_predictor.parameters(), "lr": 2 * self.hparams.initial_learning_rate},  # Duration predictor
    #         {"params": self.backbone.parameters(), "lr": self.hparams.initial_learning_rate},  # Decoder (backbone)
    #         {"params": self.head.parameters(), "lr": self.hparams.initial_learning_rate},  # Decoder (head)
    #     ]

    #     opt_disc = torch.optim.AdamW(disc_params)
    #     opt_gen = torch.optim.AdamW(gen_params)

    #     # 定义按epoch衰减学习率的lambda函数
    #     def lr_lambda(epoch):
    #         if epoch < 40:
    #             initial_lr = 1e-4
    #             stage_epochs = 40
    #         elif epoch < 60:
    #             initial_lr = 5e-5
    #             stage_epochs = 20
    #         elif epoch < 80:
    #             initial_lr = 2e-5
    #             stage_epochs = 20
    #         elif epoch < 100:
    #             initial_lr = 1e-5
    #             stage_epochs = 20
    #         else:
    #             initial_lr = 5e-6
    #             stage_epochs = 0  # 超过200的epoch保持1e-5

    #         if stage_epochs > 0:
    #             progress = (epoch - (0 if epoch < 40 else (40 if epoch < 60 else (60 if epoch < 80 else 80)))) / stage_epochs
    #             # 使用余弦衰减公式
    #             return (1 + math.cos(math.pi * progress)) / 2 * (initial_lr / self.hparams.initial_learning_rate)
    #         else:
    #             return 1e-5 / self.hparams.initial_learning_rate  # 超过200的epoch时的学习率

    #     scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda)
    #     scheduler_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)

    #     return (
    #         [opt_disc, opt_gen],
    #         [
    #             {"scheduler": scheduler_disc, "interval": "epoch"},
    #             {"scheduler": scheduler_gen, "interval": "epoch"},
    #         ],
    #     )


    def forward(self, audio_input, **kwargs):
        features, _, commit_loss = self.feature_extractor(audio_input, **kwargs)
        # print('1111', self.feature_extractor.state_dict()['encodec.decoder.model.3.convtr.convtr.weight_g'])
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output, commit_loss


    def forward_clustering(self, audio_input, **kwargs):
        features, _, commit_loss, cluster_lengths, clustered_mask = self.feature_extractor.forward_clustering(audio_input, **kwargs)
        assert cluster_lengths.dim() == 2 and torch.all(cluster_lengths.sum(dim=1) == 225), \
            "Each batch in cluster_lengths must sum to 225 and tensor must be 2D"
        # pdb.set_trace()

        avg_token = (
            cluster_lengths.ne(0).sum(dim=1).float() / 
            cluster_lengths.sum(dim=1).float()
        ).mean() * 75 * 2
        # pdb.set_trace()

        # # FastSpeech2 style duration predictor
        # log_target = torch.log(cluster_lengths.float() + 1)
        # log_pred = self.duration_predictor(features, clustered_mask)
        # # duration_loss = F.mse_loss(log_pred, log_target, reduction="mean")
        # duration_loss = F.smooth_l1_loss(log_pred, log_target)
        # duration_loss = duration_loss.masked_select(clustered_mask).mean()
        # self.log("duration/fastspeech_duration_loss", duration_loss)

        # pred_durations = torch.clamp(
        #     (torch.round(torch.exp(log_pred) - 1)),
        #     min=0,
        # )
        # pred_durations = pred_durations * clustered_mask
        # diff_ratio = (pred_durations.sum() - cluster_lengths.sum()) / cluster_lengths.sum()
        # duration_consistency_loss = diff_ratio ** 2
        # self.log("duration/duration_consistency_loss", duration_consistency_loss)

        # final_durations, use_gt, p_pred = self.duration_scheduler.sample_duration(self.global_step, cluster_lengths, pred_durations)

        # self.log("duration/use_gt", use_gt)
        # self.log("duration/p_pred", p_pred, prog_bar=True)
        # self.log("duration/pred", pred_durations[0].sum())
        # self.log("duration/gt", cluster_lengths[0].sum())

        # final_durations = pred_durations
        features, pred_len = self.length_regulator(features, cluster_lengths, max_len=225)
        # self.log("duration/pred_len_mean", pred_len.float().mean())
        # self.log("duration/pred_len_std", pred_len.float().std())
        # pdb.set_trace()

        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)

        # duration_loss += duration_consistency_loss
        duration_loss = 0.0
        return audio_output, commit_loss, avg_token, duration_loss, None


    def forward_clustering_vq(self, audio_input, **kwargs):
        features, codes, commit_loss, cluster_lengths, clustered_mask = self.feature_extractor.forward_clustering_vq(audio_input, **kwargs)
        assert cluster_lengths.dim() == 2 and torch.all(cluster_lengths.sum(dim=1) == 225), \
            "Each batch in cluster_lengths must sum to 225 and tensor must be 2D"
        # pdb.set_trace()

        avg_token = (
            cluster_lengths.ne(0).sum(dim=1).float() / 
            cluster_lengths.sum(dim=1).float()
        ).mean() * 75 * 2
        # pdb.set_trace()

        # # FastSpeech2 style duration predictor
        # # log_target = torch.log(cluster_lengths.float() + 1)
        # log_target = cluster_lengths.float()
        # log_pred = self.duration_predictor(features, clustered_mask)
        # # duration_loss = F.mse_loss(log_pred, log_target, reduction="mean")
        # # duration_loss = F.smooth_l1_loss(log_pred, log_target, reduction="none")
        # duration_loss = F.l1_loss(log_pred, log_target, reduction="none")
        # duration_loss = duration_loss.masked_select(clustered_mask).mean()
        # self.log("duration/fastspeech_duration_loss", duration_loss)
        # # pdb.set_trace()

        # # pred_durations = torch.clamp(
        # #     (torch.round(torch.exp(log_pred) - 1)),
        # #     min=0,
        # # )
        # pred_durations = torch.clamp(torch.round(log_pred), min=0.0, max=10.0)
        # pred_durations = pred_durations * clustered_mask
        # # pdb.set_trace()
        # diff_ratio = (pred_durations.sum() - cluster_lengths.sum()) / cluster_lengths.sum()
        # duration_consistency_loss = diff_ratio ** 2
        # self.log("duration/duration_consistency_loss", duration_consistency_loss)

        # final_durations, use_gt, p_pred = self.duration_scheduler.sample_duration(self.global_step, cluster_lengths, pred_durations)

        # self.log("duration/use_gt", use_gt)
        # self.log("duration/p_pred", p_pred, prog_bar=True)
        # self.log("duration/pred", pred_durations[0].sum())
        # self.log("duration/gt", cluster_lengths[0].sum())

        features, pred_len = self.length_regulator(features, cluster_lengths, max_len=225)
        # pdb.set_trace()
        # self.log("duration/pred_len_mean", pred_len.float().mean())
        # self.log("duration/pred_len_std", pred_len.float().std())
        
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)

        # VQ codebook usage
        # 注意：codes 应该是 [B, T]，来自 VQ 输出
        codes = codes.squeeze(0)
        with torch.no_grad():
            codebook_usage = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.compute_code_usage(codes)
            # codebook_usage = self.feature_extractor.simvq_quantizer.compute_code_usage(codes)
        # pdb.set_trace()

        # duration_loss += 100 * duration_consistency_loss
        duration_loss = 0.0
        return audio_output, commit_loss, avg_token, duration_loss, codebook_usage

    def forward_clustering_vq_dur(self, audio_input, **kwargs):
        features, codes, commit_loss, cluster_lengths, clustered_mask = self.feature_extractor.forward_clustering_vq(audio_input, **kwargs)
        assert cluster_lengths.dim() == 2 and torch.all(cluster_lengths.sum(dim=1) == 225), \
            "Each batch in cluster_lengths must sum to 225 and tensor must be 2D"
        # pdb.set_trace()

        avg_token = (
            cluster_lengths.ne(0).sum(dim=1).float() / 
            cluster_lengths.sum(dim=1).float()
        ).mean() * 75 * 2
        # pdb.set_trace()

        # # FastSpeech2 style duration predictor
        # # log_target = torch.log(cluster_lengths.float() + 1)
        # log_target = cluster_lengths.float()
        # log_pred = self.duration_predictor(features, clustered_mask)
        # # duration_loss = F.mse_loss(log_pred, log_target, reduction="mean")
        # # duration_loss = F.smooth_l1_loss(log_pred, log_target, reduction="none")
        # duration_loss = F.l1_loss(log_pred, log_target, reduction="none")
        # duration_loss = duration_loss.masked_select(clustered_mask).mean()
        # self.log("duration/fastspeech_duration_loss", duration_loss)
        # # pdb.set_trace()

        # # pred_durations = torch.clamp(
        # #     (torch.round(torch.exp(log_pred) - 1)),
        # #     min=0,
        # # )
        # pred_durations = torch.clamp(torch.round(log_pred), min=0.0, max=10.0)
        # pred_durations = pred_durations * clustered_mask
        # # pdb.set_trace()
        # diff_ratio = (pred_durations.sum() - cluster_lengths.sum()) / cluster_lengths.sum()
        # duration_consistency_loss = diff_ratio ** 2
        # self.log("duration/duration_consistency_loss", duration_consistency_loss)

        # final_durations, use_gt, p_pred = self.duration_scheduler.sample_duration(self.global_step, cluster_lengths, pred_durations)

        # self.log("duration/use_gt", use_gt)
        # self.log("duration/p_pred", p_pred, prog_bar=True)
        # self.log("duration/pred", pred_durations[0].sum())
        # self.log("duration/gt", cluster_lengths[0].sum())

        ## duration predictor
        if self.duration_predictor.training:
            # If in training mode, use duration_predictor to get predicted durations (logits)
            logits = self.duration_predictor(features)  # [B, T', S] -> Predicted class logits for duration
        
            # Preprocess the cluster lengths (true durations)
            _cluster_lengths = cluster_lengths - 1  # Convert cluster_lengths from [1, 4] to [0, 3]
            
            # Create mask to ignore positions where _cluster_lengths == -1
            valid_mask = (_cluster_lengths >= 0).long().view(-1)  # Mask out positions where _cluster_lengths == -1
            
            # Flatten logits, cluster_lengths, and mask for CrossEntropyLoss
            logits_flat = logits.view(-1, logits.size(-1))  # [B * T', S]
            cluster_lengths_flat = _cluster_lengths.view(-1)  # [B * T']
            clustered_mask_flat = clustered_mask.view(-1)  # [B * T']
            
            # Filter out invalid positions using the mask
            valid_positions = valid_mask.bool() & clustered_mask_flat.bool()  # Only valid and non-padded positions
            logits_flat = logits_flat[valid_positions]  # Filtered logits
            cluster_lengths_flat = cluster_lengths_flat[valid_positions]  # Filtered ground truth labels
            
            # Use CrossEntropyLoss with no reduction
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_token = criterion(logits_flat, cluster_lengths_flat)
            
            # Average the loss over valid tokens
            duration_loss = loss_per_token.sum() / valid_positions.sum()  # Normalize by the number of valid tokens

            features, pred_len = self.length_regulator(features, cluster_lengths, max_len=225)

        else:
            # If in training mode, use duration_predictor to get predicted durations (logits)
            logits, predicted_durations = self.duration_predictor.inference(features)  # [B, T', S] -> Predicted class logits for duration
        
            # Preprocess the cluster lengths (true durations)
            _cluster_lengths = cluster_lengths - 1  # Convert cluster_lengths from [1, 4] to [0, 3]
            
            # Create mask to ignore positions where _cluster_lengths == -1
            # pdb.set_trace()
            valid_mask = (_cluster_lengths >= 0).long().view(-1)  # Mask out positions where _cluster_lengths == -1
            
            # Flatten logits, cluster_lengths, and mask for CrossEntropyLoss
            logits_flat = logits.view(-1, logits.size(-1))  # [B * T', S]
            cluster_lengths_flat = _cluster_lengths.view(-1)  # [B * T']
            clustered_mask_flat = clustered_mask.view(-1)  # [B * T']
            
            # Filter out invalid positions using the mask
            valid_positions = valid_mask.bool() & clustered_mask_flat.bool()  # Only valid and non-padded positions
            logits_flat = logits_flat[valid_positions]  # Filtered logits
            cluster_lengths_flat = cluster_lengths_flat[valid_positions]  # Filtered ground truth labels
            
            # Use CrossEntropyLoss with no reduction
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_token = criterion(logits_flat, cluster_lengths_flat)
            
            # Average the loss over valid tokens
            duration_loss = loss_per_token.sum() / valid_positions.sum()  # Normalize by the number of valid tokens
            
            features, pred_len = self.length_regulator(features, predicted_durations, max_len=225)

        self.log("duration/duration_loss", duration_loss.item(), prog_bar=True)
            
        # pdb.set_trace()
        # self.log("duration/pred_len_mean", pred_len.float().mean())
        # self.log("duration/pred_len_std", pred_len.float().std())
        
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)

        # VQ codebook usage
        # 注意：codes 应该是 [B, T]，来自 VQ 输出
        codes = codes.squeeze(0)
        with torch.no_grad():
            codebook_usage = self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.compute_code_usage(codes)
            # codebook_usage = self.feature_extractor.simvq_quantizer.compute_code_usage(codes)
        # pdb.set_trace()

        # duration_loss += 100 * duration_consistency_loss
        # duration_loss = 0.0
        return audio_output, commit_loss, avg_token, duration_loss, codebook_usage


    def forward_encoder_recon(self, audio_input, **kwargs):
        audio_embeddings = self.feature_extractor.forward_encoder(audio_input, **kwargs)
        x = self.backbone(audio_embeddings, **kwargs)
        audio_output = self.head(x)
        return audio_output

    # def optimizer_step(self, *args, **kwargs):
    #     # 这里已经loss.backward()了！！！
    #     for n, p in self.feature_extractor.simvq_quantizer.named_parameters():
    #         print(n, p.grad)
    #         if p.grad is not None:
    #             print(f"Grad {n}: mean={p.grad.abs().mean().item()}")
    #     super().optimizer_step(*args, **kwargs)


    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input = batch

        # train discriminator
        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                audio_hat, _, _, _, _ = self.forward_clustering_vq(audio_input, **kwargs)
                # audio_hat, _, _, _, _ = self.forward_clustering_vq_dur(audio_input, **kwargs)

            loss_dac=self.dacdiscriminator.discriminator_loss(audio_hat.unsqueeze(1),audio_input.unsqueeze(1))

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd + loss_dac

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            self.log("discriminator/dac", loss_dac)
            return loss

        # train generator
        if optimizer_idx == 1:
            audio_hat, commit_loss, avg_token, duration_loss, codebook_usage = self.forward_clustering_vq(audio_input, **kwargs)
            # audio_hat, commit_loss, avg_token, duration_loss, codebook_usage = self.forward_clustering_vq_dur(audio_input, **kwargs)
            if self.train_discriminator:

                loss_dac_1,loss_dac_2 = self.dacdiscriminator.generator_loss(audio_hat.unsqueeze(1),audio_input.unsqueeze(1))
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
                self.log("generator/loss_dac_1", loss_dac_1)
                self.log("generator/loss_dac_2", loss_dac_2)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = loss_dac_1 = loss_dac_2 = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
                + 1000 * commit_loss
                + loss_dac_1
                + loss_dac_2
            )
            mel_loss_ = 0.0
            # audio_hat_ = self.forward_encoder_recon(audio_input, **kwargs)
            # mel_loss_ = self.melspec_loss(audio_hat_, audio_input)
            loss += self.mel_loss_coeff * mel_loss_
            # loss += 70 * duration_loss

            ###### duration loss
            # loss = duration_loss

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("mel_loss_coeff", self.mel_loss_coeff)
            self.log("generator/mel_loss", mel_loss, prog_bar=True)
            self.log("commit_loss", commit_loss, prog_bar=True)
            self.log("avg_token", avg_token, prog_bar=True)
            # self.log("generator/mel_loss_enc_dec", mel_loss_)
            # self.log("duration/duration_loss", duration_loss, prog_bar=True)
            # self.log("generator/duration_loss", duration_loss, prog_bar=True)

            # VQ codebook usage
            if codebook_usage is not None:
                num_active_codes = (codebook_usage > 0).sum().item()
                usage_percent = num_active_codes / self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.codebook_size
                self.log("vq/codebook_usage_percent", usage_percent, prog_bar=True)
                topk = torch.topk(codebook_usage, 10).indices
                lowk = (codebook_usage == 0).nonzero(as_tuple=False).flatten()
                self.log("vq/top1_used_code", topk[0].item())
                self.log("vq/num_dead_codes", len(lowk))


            if self.global_step % 1000 == 0 and self.global_rank == 0:
                self.logger.experiment.add_audio(
                    "train/audio_in", self.prepare_audio(audio_input[0]), self.global_step, self.hparams.sample_rate
                )
                self.logger.experiment.add_audio(
                    "train/audio_pred", self.prepare_audio(audio_hat[0]), self.global_step, self.hparams.sample_rate
                )
                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
                # pdb.set_trace()
                self.logger.experiment.add_image(
                    "train/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "train/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.flush()

            return loss

    def on_validation_epoch_start(self):
        if self.hparams.evaluate_utmos:
            from metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        audio_hat, commit_loss, avg_token, duration_loss, codebook_usage = self.forward_clustering_vq(audio_input, **kwargs)
        # audio_hat, commit_loss, avg_token, duration_loss, codebook_usage = self.forward_clustering_vq_dur(audio_input, **kwargs)

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        if self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        # total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score) + 1000 * commit_loss
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score)
        # print(avg_token)

        # VQ codebook usage
        if codebook_usage is not None:
            num_active_codes = (codebook_usage > 0).sum().item()
            usage_percent = num_active_codes / self.feature_extractor.encodec.quantizer.vq.layers[0]._codebook.codebook_size
        else:
            usage_percent = 0.0

        return {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "avg_token": avg_token,
            "duration_loss": duration_loss,
            "codebook_usage_percent": usage_percent,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }

    
    # def prepare_audio(self, audio):
    #     # 转为 float32 + 归一化（如果没归一化的话）
    #     audio_np = audio.data.cpu().float().numpy()
    #     if audio_np.ndim == 1:
    #         audio_np = audio_np[None, :]  # (1, num_samples)
    #     elif audio_np.ndim == 2 and audio_np.shape[0] != 1:
    #         audio_np = audio_np[:1, :]  # 取前一条
    #     return audio_np.astype("float32")

    def prepare_audio(self, audio):
        # audio: torch.Tensor, shape: [num_samples] or [B, num_samples]
        audio = audio.data.cpu().float()
        audio = audio / (audio.abs().max() + 1e-9)  # 避免爆音
        audio_np = audio.squeeze().numpy()  # 保证是一维

        if audio_np.ndim == 1:
            audio_np = np.expand_dims(audio_np, axis=1)  # (num_samples, 1)

        return audio_np.astype("float32")


    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, audio_pred = outputs[0].values()
            # pdb.set_trace()
            self.logger.experiment.add_audio(
                "val_in", self.prepare_audio(audio_in), self.global_step, self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "val_pred", self.prepare_audio(audio_pred), self.global_step, self.hparams.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            # pdb.set_trace()
            self.logger.experiment.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.flush()

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()
        avg_token = np.array([x["avg_token"].cpu().item() for x in outputs]).mean()
        # duration_loss = np.array([x["duration_loss"].cpu().item() for x in outputs]).mean()
        codebook_usage_percent = np.array([x["codebook_usage_percent"] for x in outputs]).mean()
        # pdb.set_trace()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)
        # self.log("val/duration_loss", duration_loss, sync_dist=True)
        self.log("val/codebook_usage_percent", codebook_usage_percent, sync_dist=True)
        self.log("val/avg_token", avg_token, sync_dist=True)

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

        # self.train_discriminator = False

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # for param in self.head.parameters():
        #     param.requires_grad = False
        # for param in self.duration_predictor.parameters():
        #     param.requires_grad = True
        

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


class WavTokenizer(VocosExp):
    """
    WavTokenizer is a subclass of VocosExp that overrides the parent experiment to function as a conditional GAN.
    It manages an additional `bandwidth_id` attribute, which denotes a learnable embedding corresponding to
    a specific bandwidth value of EnCodec. During training, a random bandwidth_id is generated for each step,
    while during validation, a fixed bandwidth_id is used.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        resume_config: str,
        resume_model: str,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        resume: bool = False,
    ):
        super().__init__(
            feature_extractor,
            backbone,
            head,
            resume_config,
            resume_model,
            sample_rate,
            initial_learning_rate,
            num_warmup_steps,
            mel_loss_coeff,
            mrd_loss_coeff,
            pretrain_mel_steps,
            decay_mel_coeff,
            evaluate_utmos,
            evaluate_pesq,
            evaluate_periodicty,
            resume
        )
        # Override with conditional discriminators
        # VocosExp.__init__(self, feature_extractor, backbone, head, resume_config, resume_model)
        # if self.resume:
        #     VocosExp.load_from_checkpoint(self.resume_model)
        self.multiperioddisc = MultiPeriodDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.multiresddisc = MultiResolutionDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.dac = DACDiscriminator()
        if self.resume:
            print('加载预训练模型:', self.resume_model)
            
            state_dict_raw = torch.load(self.resume_model, map_location=self.device)['state_dict']
            state_dict_fa_qa = dict()
            state_dict_fa_en = dict()
            state_dict_fa_de = dict()
            state_dict_bb = dict()
            state_dict_hd = dict()
            state_dict_mp = dict()
            state_dict_mr = dict()
            state_dict_dac = dict()
            state_dict_dr = dict()
            for k, v in state_dict_raw.items():
                # breakpoint()
                if k.startswith('feature_extractor.encodec.quantizer'):
                    # breakpoint()
                    # print("*****",k)
                    ss = k[46:48]
                    if ss[-1] == '.':
                        num = int(ss[0])
                        # print("num,k",num,k[36:])
                        if num <= 7:
                            state_dict_fa_qa[k[36:]] = v
                if k.startswith('feature_extractor.encodec.encoder'):
                    state_dict_fa_en[k[34:]] = v
                if k.startswith('feature_extractor.encodec.decoder'):
                    state_dict_fa_de[k[34:]] = v
                if k.startswith('backbone.'):
                    state_dict_bb[k[9:]] = v
                if k.startswith('head.'):
                    state_dict_hd[k[5:]] = v
                if k.startswith('multiperioddisc.'):
                    state_dict_mp[k[16:]] = v
                if k.startswith('multiresddisc.'):
                    state_dict_mr[k[14:]] = v
                if k.startswith('dac.'):
                    state_dict_dac[k[4:]] = v
                # if k.startswith('duration_predictor.'):
                #     state_dict_dr[k[19:]] = v
            # breakpoint()
            # 是否加载量化器部分权重
            feature_extractor.encodec.quantizer.load_state_dict(state_dict_fa_qa, strict=True)
            feature_extractor.encodec.encoder.load_state_dict(state_dict_fa_en, strict=True)
            feature_extractor.encodec.decoder.load_state_dict(state_dict_fa_de, strict=True)
            # feature_extractor.encodec.quantizer.load_state_dict(state_dict_fa_qa, strict=True)
            backbone.load_state_dict(state_dict_bb, strict=True)
            head.load_state_dict(state_dict_hd, strict=True)
            self.feature_extractor = feature_extractor.to(self.device)
            self.backbone = backbone.to(self.device)
            self.head = head.to(self.device)
            self.multiperioddisc.load_state_dict(state_dict_mp, strict=True)
            self.multiresddisc.load_state_dict(state_dict_mr, strict=True)
            self.dac.load_state_dict(state_dict_dac, strict=True)
            # self.duration_predictor.load_state_dict(state_dict_dr, strict=True)

            # 固定encoder权重，促进VQ和duration predictor的学习
            # for param in self.feature_extractor.encodec.encoder.parameters():
            #     param.requires_grad = False
            # for param in self.feature_extractor.encodec.decoder.parameters():
            #     param.requires_grad = False
            # for param in self.backbone.parameters():
            #     param.requires_grad = False
            # for param in self.head.parameters():
            #     param.requires_grad = False

    def training_step(self, *args):
        # print('-------------------train--------------------')
        # if self.global_rank == 0 and self.resume:
        #     config_path = self.resume_config
        #     model_path = self.resume_model
        #     self.pretrained_load(config_path, model_path)
        #     print('加载预训练模型:', model_path)
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device,)
        output = super().training_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_step(self, *args):
        # print('-------------------valid--------------------')
        bandwidth_id = torch.tensor([0], device=self.device)
        output = super().validation_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, _ = outputs[0].values()
            # Resynthesis with encodec for reference
            self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[0])
            encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
            self.logger.experiment.add_audio(
                "encodec", encodec_audio[0, 0].data.cpu().numpy(), self.global_step, self.hparams.sample_rate,
            )

        super().validation_epoch_end(outputs)

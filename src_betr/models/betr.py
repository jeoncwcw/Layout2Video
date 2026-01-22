import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from models.feature_modules import FeatureGenerator, DA3FeatureExtractor, DINOv3FeatureExtractor
from models.transformer_modules import (
    TransformerEncoder, TransformerDecoder, 
    build_position_encoding, BoxEmbedding, _prepare_mask_for_transformer, _unpatchify,
)
from models.dense_modules import DenseHeads, UpsampleLayer, SoftArgmax2D

from models.transformer_modules.encoder import TransformerEncoderLayer
from models.transformer_modules.decoder import TransformerDecoderLayer
import torch
from torch import nn

class BETRModel(nn.Module):
    def __init__(self, cfg):
        super(BETRModel, self).__init__()
        # Feature Extractors
        self.feature_mode = cfg.feature_mode
        if not self.feature_mode:
            print("[Image Mode] Loading Backbones ...")
            self.feature_monodepth = DA3FeatureExtractor(
                cfg_path=Path(cfg.monodepth_cfg_path),
                checkpoint_path=Path(cfg.monodepth_checkpoint_path),
                device=cfg.device,
            )
            self.feature_metricdepth = DA3FeatureExtractor(
                cfg_path=Path(cfg.metricdepth_cfg_path),
                checkpoint_path=Path(cfg.metricdepth_checkpoint_path),
                device=cfg.device,
            )
            self.feature_dinov3 = DINOv3FeatureExtractor(
                checkpoint_path=Path(cfg.dinov3_checkpoint_path),
                device=cfg.device,
            )
            self.feature_dropout = nn.Dropout2d(p = cfg.aug.feature_dropout)
        else:
            print("[Feature Mode] Skipping Backbone Loading ...")
            self.feature_monodepth = None
            self.feature_metricdepth = None
            self.feature_dinov3 = None
            
        # Feature Generator
        self.feature_generator = FeatureGenerator(cfg.feature_generator_dim)
        in_channels = (cfg.feature_generator_dim * 2) + 1024  # Assuming DINOv3 outputs 1024 channels
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=cfg.d_model, kernel_size=1)
        self.skip_projection = nn.Conv2d(in_channels=in_channels, out_channels=cfg.skip_channels, kernel_size=1)

        
        # Transformer Encoder and Decoder
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.encoder.nhead,
            dim_feedforward=cfg.encoder.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, cfg.encoder.num_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.decoder.nhead,
            dim_feedforward=cfg.decoder.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, cfg.decoder.num_layers)

        # Positional Encoding and Box Embedding
        self.position_encoding = build_position_encoding(
            hidden_dim=cfg.d_model,
            temperature= cfg.position_encoding.temperature,
            normalize=cfg.position_encoding.normalize,
            scale=cfg.position_encoding.scale,
        )
        self.box_embedding = BoxEmbedding(
            d_model=cfg.d_model,
            temperature=cfg.box_embedding.temperature,
            scale=cfg.box_embedding.scale,
        )

        # Upsample and Prediction Heads
        self.upsample = UpsampleLayer(d_model=cfg.d_model, skip_channels=cfg.skip_channels, activation=cfg.activation)
        self.prediction_heads = DenseHeads(
            heads=cfg.prediction_heads,
            in_channels=cfg.d_model // 4,  # After two upsampling layers
        )
        self.soft_argmax = SoftArgmax2D(beta=cfg.soft_argmax.beta, is_sigmoid=cfg.soft_argmax.is_sigmoid)
        
        
        # # Augmentation parameters
        self.box_jitter_sigma = cfg.aug.box_jitter_sigma
        self.feature_noise_sigma = cfg.aug.feature_noise_sigma
        self.feature_dropout = nn.Dropout2d(p = cfg.aug.feature_dropout)

    def forward(self, bbx2d_tight, mask = None,
                images_da3 = None, images_dino = None,
                f_metric = None, f_mono = None, f_dino = None,
                ):
        # Generate combined features
        try:
            if self.feature_mode:
                metric_depth, mono_depth, dinov3_features = f_metric, f_mono, f_dino
            else:
                self.feature_monodepth.model.eval()
                self.feature_metricdepth.model.eval()
                self.feature_dinov3.model.eval()
                with torch.no_grad():
                    metric_depth = self.feature_metricdepth(images_da3)
                    mono_depth = self.feature_monodepth(images_da3)
                    dinov3_features = self.feature_dinov3(images_dino)
        except RuntimeError as e:
            print("RuntimeError in feature extraction:", e)
            raise e
        # if self.training and self.feature_noise_sigma > 0:
        #     noise = torch.randn_like(bbx2d_tight) * self.box_jitter_sigma
        #     bbx2d_tight = torch.clamp(bbx2d_tight + noise, 0, 1)
        #     metric_depth = metric_depth + torch.randn_like(metric_depth) * self.feature_noise_sigma
        #     mono_depth = mono_depth + torch.randn_like(mono_depth) * self.feature_noise_sigma
        #     dinov3_features = dinov3_features + torch.randn_like(dinov3_features) * self.feature_noise_sigma     
        combined_features = self.feature_generator(metric_depth, mono_depth, dinov3_features) # [B, ]
        skip_feature = self.skip_projection(combined_features)  # For potential skip connections
        combined_features = self.feature_dropout(combined_features)
        
        # Add positional encoding
        combined_features = self.conv1x1(combined_features)
        pos_encoding = self.position_encoding(combined_features)
        padding_mask = _prepare_mask_for_transformer(mask)
        box_embeddings = self.box_embedding(bbx2d_tight)

        # Permuting for transformer input
        combined_features = combined_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)    
        pos_encoding = pos_encoding.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        box_embeddings = box_embeddings.permute(1, 0, 2)  # (4, B, C)
        
        # Pass through Transformer and unpatchify
        image_feat = self.transformer_encoder(combined_features, image_feat_key_padding_mask=padding_mask, pos=pos_encoding)
        output = self.transformer_decoder(image_feat, box_embeddings, image_feat_key_padding_mask=padding_mask, image_feat_pos=pos_encoding)

        output = _unpatchify(output) # (B, C, H, W)

        # Upsample, Prediction Heads, Soft Argmax and get center coords
        output = self.upsample(output, skip_feature)
        output = self.prediction_heads(output)
        corner_heatmaps = output['corner heatmaps'] # [B, 8, 128, 128]
        corner_coords = self.soft_argmax(corner_heatmaps) # [B, 8, 2]
        corner_coords_orig = corner_coords*4
        output['corner coords'] = corner_coords_orig

        return output
    
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


from torch import nn

class BETRModel(nn.Module):
    def __init__(self, cfg):
        super(BETRModel, self).__init__()
        # Feature Extractors
        self.feature_mode = cfg.feature_mode
        if not self.feature_mode:
            print("🚀 Loading Heavy Backbones (Image Training Mode)...")
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
        else:
            print("⚡ Feature Mode On: Skipping Backbone Loading (Memory Saved!)")
            self.feature_monodepth = None
            self.feature_metricdepth = None
            self.feature_dinov3 = None

        # Feature Generator
        self.feature_generator = FeatureGenerator(cfg.feature_generator_dim)
        in_channels = (cfg.feature_generator_dim * 2) + 1024  # Assuming DINOv3 outputs 1024 channels
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=cfg.d_model, kernel_size=1)

        
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
        self.upsample = UpsampleLayer(d_model=cfg.d_model, activation=cfg.activation)
        self.prediction_heads = DenseHeads(
            heads=cfg.prediction_heads,
            in_channels=cfg.d_model // 4,  # After two upsampling layers
        )
        self.soft_argmax = SoftArgmax2D(beta=cfg.soft_argmax.beta, is_sigmoid=cfg.soft_argmax.is_sigmoid)

    def forward(self, bbx2d_tight, mask = None,
                images_da3 = None, images_dino = None,
                f_metric = None, f_mono = None, f_dino = None,
                feature_mode = False):
        # Generate combined features
        try:
            if feature_mode:
                metric_depth, mono_depth, dinov3_features = f_metric, f_mono, f_dino
            else:
                metric_depth = self.feature_metricdepth(images_da3)
                mono_depth = self.feature_monodepth(images_da3)
                dinov3_features = self.feature_dinov3(images_dino)
        except RuntimeError as e:
            print("RuntimeError in feature extraction:", e)
            raise e
        combined_features = self.feature_generator(metric_depth, mono_depth, dinov3_features)

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
        output = self.upsample(output)
        output = self.prediction_heads(output)
        center_heatmap = output['center heatmap'] # [B, 1, 128, 128]
        center_coords = self.soft_argmax(center_heatmap) # [B, 1, 2]
        center_coords_orig = center_coords*4
        output['center coords'] = center_coords_orig

        return output
    
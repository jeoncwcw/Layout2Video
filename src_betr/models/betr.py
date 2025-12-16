import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from models.feature_modules import FeatureGenerator, DA3FeatureExtractor, DINOv3FeatureExtractor
from models.transformer_modules import TransformerEncoder, TransformerDecoder, build_position_encoding, BoxEmbedding, _prepare_mask_for_transformer

from torch import nn

class BETRModel(nn.Module):
    def __init__(self, cfg):
        super(BETRModel, self).__init__()
        self.feature_monodepth = DA3FeatureExtractor(
            cfg_path=cfg.monodepth_cfg_path,
            checkpoint_path=cfg.monodepth_checkpoint_path,
            device=cfg.device,
        )
        self.feature_metricdepth = DA3FeatureExtractor(
            cfg_path=cfg.metricdepth_cfg_path,
            checkpoint_path=cfg.metricdepth_checkpoint_path,
            device=cfg.device,
        )
        self.feature_dinov3 = DINOv3FeatureExtractor(
            checkpoint_path=cfg.dinov3_checkpoint_path,
            device=cfg.device,
        )
        self.conv1x1 = nn.Conv2d(in_channels=cfg.feature_generator*4, out_channels=cfg.encoder.d_model, kernel_size=1)
        self.feature_generator = FeatureGenerator(cfg.feature_generator)
        self.transformer_encoder = TransformerEncoder(cfg.encoder)
        self.transformer_decoder = TransformerDecoder(cfg.decoder)
        self.position_encoding = build_position_encoding(cfg.position_encoding)
        self.box_embedding = BoxEmbedding(cfg.box_embedding)

    def forward(self, images_da3, images_dino, bbx2d_tight, mask = None):
        # Generate combined features
        metric_depth = self.feature_metricdepth(images_da3)
        mono_depth = self.feature_monodepth(images_da3)
        dinov3_features = self.feature_dinov3(images_dino)
        combined_features = self.feature_generator(metric_depth, mono_depth, dinov3_features)

        # Add positional encoding
        combined_features = self.conv1x1(combined_features)
        pos_encoding = self.position_encoding(combined_features)
        padding_mask = _prepare_mask_for_transformer(mask)
        box_embeddings = self.box_embedding(bbx2d_tight)

        # Pass through encoder
        memory = self.transformer_encoder(combined_features, image_feat_key_padding_mask=padding_mask, pos=pos_encoding)

        # Pass through decoder (assuming some target input is provided)
        output = self.transformer_decoder(memory, box_embeddings, image_feat_key_padding_mask=padding_mask, image_feat_pos=pos_encoding)

        return output
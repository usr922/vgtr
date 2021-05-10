import torch.nn as nn
from .vg_encoder import VGEncoder
from .vg_decoder import VGDecoder
from .position_encoding import PositionEmbeddingSine


class VGTR(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.input_proj = nn.Conv2d(2048, args.hidden_dim, kernel_size=1)

        self.encoder = VGEncoder(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers)

        self.decoder = VGDecoder(n_layers=args.dec_layers,
                                 n_heads=args.nheads,
                                 d_model=args.hidden_dim)

        self.pos_encoder = PositionEmbeddingSine(args.hidden_dim // 2, normalize=False)

    def forward(self, img, sent, sent_id):

        pos_feature = self.pos_encoder(img)

        # encoder
        fused_vis_feature, fused_exp_feature = self.encoder(self.input_proj(img), pos_feature, sent)

        # decoder
        out = self.decoder(fused_vis_feature.transpose(0, 1), fused_exp_feature,
                           pos_feature=pos_feature.flatten(2).permute(2, 0, 1))

        return out.transpose(0, 1)


def build_vgtr(args):

    return VGTR(args)
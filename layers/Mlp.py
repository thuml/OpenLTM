import torch
import torch.nn as nn
from layers.SelfAttention_Family import TTMGatedAttention


class TTMmlp(nn.Module):
    def __init__(self, in_features, out_features, e_factor, dropout):
        super().__init__()
        num_hidden = in_features * e_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TTMMixerBlock(nn.Module):
    def __init__(self, d_model, features, mode, dropout):
        super().__init__()

        self.mode = mode

        self.norm = nn.LayerNorm(d_model)

        self.mlp = TTMmlp(
            in_features=features,
            out_features=features,
            e_factor=2,
            dropout=dropout,
        )

        self.gating_block = TTMGatedAttention(in_size=features, out_size=features)

    def forward(self, x):
        residual = x  # [B M N P]
        x = self.norm(x)

        assert self.mode in ["patch", "feature", "channel"]

        if self.mode == "patch":
            x = x.permute(0, 1, 3, 2)  # [B M P N]
        elif self.mode == "channel":
            x = x.permute(0, 3, 2, 1)  # [B P N M]
        else:
            pass

        x = self.mlp(x)
        x = self.gating_block(x)

        if self.mode == "patch":
            x = x.permute(0, 1, 3, 2)  # [B M N P]
        elif self.mode == "channel":
            x = x.permute(0, 3, 2, 1)  # [B M N P]
        else:
            pass

        out = x + residual
        return out


class TTMLayer(nn.Module):
    def __init__(self, d_model, num_patches, n_vars, mode, dropout):
        super().__init__()

        if num_patches > 1:
            self.patch_mixer = TTMMixerBlock(
                d_model=d_model, features=num_patches, mode="patch", dropout=dropout
            )

        self.feature_mixer = TTMMixerBlock(
            d_model=d_model, features=d_model, mode="feature", dropout=dropout
        )

        self.mode = mode
        self.num_patches = num_patches
        if self.mode == "mix_channel":
            self.channel_feature_mixer = TTMMixerBlock(
                d_model=d_model, features=n_vars, mode="channel", dropout=dropout
            )

    def forward(self, x):
        if self.mode == "mix_channel":
            x = self.channel_feature_mixer(x)  # [B M N P]

        if self.num_patches > 1:
            x = self.patch_mixer(x)  # [B M N P]

        x = self.feature_mixer(x)  # [B M N P]

        return x

import torch
import torch.nn as nn
from layers.MLP import TTMLayer


class Model(nn.Module):
    """
    TTM: Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series (NeurIPS 2024)

    Paper: https://arxiv.org/pdf/2401.03955

    GitHub: https://github.com/ibm-granite/granite-tsfm

    Released Model: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1
    
    Citation: @inproceedings{Ekambaram2024TTM,
        title={Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series},
        author={Vijay Ekambaram and Arindam Jati and Pankaj Dayama and Sumanta Mukherjee and Nam H. Nguyen and Wesley M. Gifford and Chandra Reddy and Jayant Kalagnanam},
        booktitle={Neural Information Processing Systems},
        year={2024}
    }
    
    Note: This implementation is a simplified version of the original implementation. Simplify some settings and model structures.
    """

    def __init__(self, configs):
        super().__init__()

        configs.num_patches = (
            max(configs.seq_len, configs.patch_size) - configs.patch_size
        ) // configs.stride + 1

        self.configs = configs
        self.pred_len = configs.test_pred_len
        self.n_vars = configs.n_vars
        self.backbone = TTMBackbone(configs)
        self.use_decoder = configs.use_decoder

        self.use_norm = configs.use_norm

        if configs.use_decoder:
            self.decoder_adapter = nn.Linear(configs.d_model, configs.decoder_d_model) # [B M N D] -> [B M N decoder_d_model]
            self.decoder = TTMBlock(
                e_layers=configs.decoder_num_layers,
                AP_levels=0,
                d_model=configs.decoder_d_model,
                num_patches=configs.num_patches,
                n_vars=configs.n_vars,
                mode=configs.decoder_mode,
                dropout=configs.dropout,
            )

        self.head = TTMPredicationHead(configs=configs)

    def forward(
        self,
        x,
        x_mark,
        y_mark,
    ):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # x [B L M]
        decoder_input = self.backbone(x)  # [B M N D]

        if self.use_decoder:
            decoder_input = self.decoder_adapter(
                decoder_input
            )  # [B M N decoder_d_model]
            decoder_output = self.decoder(decoder_input)  # [B M N decoder_d_model]
        else:
            decoder_output = decoder_input

        y_hat = self.head(decoder_output)  # [B L M]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            y_hat = y_hat * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            y_hat = y_hat + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return y_hat

# adaptive patching block: in TTMBackbone, different TTMAPBlock operate at varying patch lengths and numbers of patches. 
# which aims to resolve multi-resolution issues in modelling diverse TS datasets.
class TTMAPBlock(nn.Module):
    def __init__(
        self,
        e_layers,
        d_model,
        num_patches,
        n_vars,
        mode,
        adapt_patch_level,
        dropout,
    ):
        """
            mode: determines how to process the channels, details in TTMLayer
            adapt_patch_level: the level of adaptive patching, determines the patch length and number of patches
        """
        super().__init__()
        self.adapt_patch_level = adapt_patch_level
        adaptive_patch_factor = 2**adapt_patch_level
        self.adaptive_patch_factor = adaptive_patch_factor

        self.mixer_layers = nn.ModuleList(
            [
                TTMLayer(
                    d_model=d_model // self.adaptive_patch_factor,
                    num_patches=num_patches * self.adaptive_patch_factor,
                    n_vars=n_vars,
                    mode=mode,
                    dropout=dropout,
                )
                for i in range(e_layers)
            ]
        )

    def forward(self, x):
        x = torch.reshape(
            x,
            (
                x.shape[0],
                x.shape[1],
                x.shape[2] * self.adaptive_patch_factor,
                x.shape[3] // self.adaptive_patch_factor,
            ),
        )

        for mod in self.mixer_layers:
            x = mod(x)

        x = torch.reshape(
            x,
            (
                x.shape[0],
                x.shape[1],
                x.shape[2] // self.adaptive_patch_factor,
                x.shape[3] * self.adaptive_patch_factor,
            ),
        )

        return x


class TTMBlock(nn.Module):
    def __init__(
        self, e_layers, AP_levels, d_model, num_patches, n_vars, mode, dropout
    ):
        """
            mode: determines how to process the channels, details in TTMLayer
            AP_level: the level of adaptive patching, determines the number of adaptive patching blocks
        """
        super().__init__()

        e_layers = e_layers

        self.AP_levels = AP_levels

        if self.AP_levels > 0:
            # different TTMAPBlock at varying patch lengths and numbers of patches
            self.mixers = nn.ModuleList(
                [
                    TTMAPBlock(
                        e_layers=e_layers,
                        d_model=d_model,
                        num_patches=num_patches,
                        n_vars=n_vars,
                        mode=mode,
                        adapt_patch_level=i,
                        dropout=dropout,
                    )
                    for i in reversed(range(self.AP_levels))
                ]
            )

        else:
            self.mixers = nn.ModuleList(
                [
                    TTMLayer(
                        d_model=d_model,
                        num_patches=num_patches,
                        n_vars=n_vars,
                        mode=mode,
                        dropout=dropout,
                    )
                    for _ in range(e_layers)
                ]
            )

    def forward(self, x):
        for mod in self.mixers:
            x = mod(x)

        return x


class TTMPredicationHead(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.dropout_layer = nn.Dropout(configs.dropout)
        if configs.use_decoder:
            head_d_model = configs.decoder_d_model
        else:
            head_d_model = configs.d_model

        self.base_forecast_block = nn.Linear(
            (configs.num_patches * head_d_model), configs.test_pred_len
        )

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):
        x = self.flatten(x)  # [B M N*D]
        x = self.dropout_layer(x)  # [B M N*D]
        output = self.base_forecast_block(x)  # [B M L]
        output = output.transpose(-1, -2)  # [B L M]

        return output


class TTMBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.encoder = TTMBlock(
            e_layers=configs.e_layers,
            AP_levels=configs.AP_levels,
            d_model=configs.d_model,
            num_patches=configs.num_patches,
            n_vars=configs.n_vars,
            mode=configs.mode,
            dropout=configs.dropout,
        )
        self.patcher = nn.Linear(configs.patch_size, configs.d_model)
        self.patch_size = configs.patch_size
        self.stride = configs.stride

    def forward(self, x):
        # past_values [B L M]
        x = x.permute(0, 2, 1)  # [B M L]
        patched_x = x.unfold(
            dimension=-1, size=self.patch_size, step=self.stride
        )  # [B M N P]
        patched_x = self.patcher(patched_x)  # [B M N D]

        encoder_output = self.encoder(patched_x)  # [B M N D]

        return encoder_output

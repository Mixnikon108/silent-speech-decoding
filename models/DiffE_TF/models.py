import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange
from utils import *

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, inc: int, outc: int, kernel_size: int, stride=1, gn=8):
        super().__init__()
        """
        standard ResNet style convolutional block
        """
        self.same_channels = inc == outc
        self.ks = kernel_size
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, get_padding(self.ks)),
            nn.GroupNorm(gn, outc),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            out = (x + x1) / 2
        else:
            out = x1
        return out


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetDown, self).__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)
        return x


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetUp, self).__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode="nearest")
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer(x)
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(ConditionalUNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        # Outputs of downsampling blocks
        self.d1_out = n_feat * 1        # after first down
        self.d2_out = n_feat * 2        # after second down

        # Outputs of upsampling blocks
        self.u1_out = n_feat             # after first up
        self.u2_out = in_channels       # final output before skip

        # Positional embedding for diffusion time
        self.sin_emb = SinusoidalPosEmb(n_feat)

        # Two downsampling blocks (removed third)
        self.down1 = UnetDown(in_channels, self.d1_out, kernel_size=1, gn=8, factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, kernel_size=1, gn=8, factor=2)

        # Two upsampling blocks corresponding to the two downsamples
        self.up1 = UnetUp(self.d2_out, self.u1_out, kernel_size=1, gn=8, factor=2)
        # Skip connection: concatenate up1 + time embed + down1
        self.up2 = UnetUp(self.u1_out + self.d1_out, self.u2_out, kernel_size=1, gn=8, factor=2)

        # Final conv to map back to in_channels, concatenating skip from input
        self.out = nn.Conv1d(self.u2_out + in_channels, in_channels, 1)

    def forward(self, x, t):
        # Downsampling path
        down1 = self.down1(x)          # e.g., 800 -> 400
        down2 = self.down2(down1)      # e.g., 400 -> 200

        # Time embedding
        temb = self.sin_emb(t).view(-1, self.n_feat, 1)

        # Upsampling path
        up1 = self.up1(down2)          # e.g., 200 -> 400
        # Inject time embedding and skip connection from down2
        up1 = up1 + temb

        # Second up: use skip from down1
        up2 = self.up2(torch.cat([up1, down1], dim=1))  # 400 -> 800

        # Final output with input skip
        out = self.out(torch.cat([up2, x], dim=1))      # 800 -> 800

        return out, (down1, down2), (up1, up2)
    

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def ddpm_schedules(beta1, beta2, T):
    # assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    beta_t = cosine_beta_schedule(T, s=0.008).float()
    # beta_t = sigmoid_beta_schedule(T).float()

    alpha_t = 1 - beta_t

    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)

    return {
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device

    def forward(self, x):
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            self.device
        )  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * noise
        times = _ts / self.n_T
        output, down, up = self.nn_model(x_t, times)
        return output, down, up, noise, times


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, latent_dim, emb_dim):
        super().__init__()
        self.linear_out = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),    # NEW
            nn.PReLU(),
            nn.Dropout(0.4),                # NEW: probabilidad 40%
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),    # NEW
            nn.PReLU(),
            nn.Dropout(0.4),                # NEW
            nn.Linear(latent_dim, emb_dim),
        )

    def forward(self, x):
        return self.linear_out(x)





# ------------------------------------------------------------------ #
# 1. Positional encoding (learnable)
# ------------------------------------------------------------------ #
class PosEmbed(nn.Module):
    """Learnable 1‑D positional embedding (kept identical to 2‑D version)."""

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, : x.size(1)]


# ------------------------------------------------------------------ #
# 2. Patch embedding (1‑D) for EEG
# ------------------------------------------------------------------ #
class PatchEmbed1D(nn.Module):
    """
    Embed a 1‑D signal into non‑overlapping patches using Conv1d.

    Args:
        in_ch (int):  # EEG channels (64).
        patch_size (int): size of each temporal patch (e.g. 25 → 32 patches for 800‑sample trials).
        dim (int): embedding dimension per patch.
    """

    def __init__(self, in_ch: int, patch_size: int, dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_ch, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = self.proj(x)                    # (B, dim, L / p)
        x = rearrange(x, "b d l -> b l d")  # (B, N, dim)
        return x


# ------------------------------------------------------------------ #
# 3. Pre‑norm Transformer block (identical to vision version)
# ------------------------------------------------------------------ #
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------------ #
# 4. ViT encoder for 1‑D EEG
# ------------------------------------------------------------------ #
class ViTEncoder1D(nn.Module):
    """Encoder that outputs (cls_token, patch_tokens)."""

    def __init__(self, *, in_ch: int, signal_len: int, patch_size: int, dim: int, depth: int, heads: int):
        super().__init__()
        assert signal_len % patch_size == 0, "signal_len debe ser múltiplo de patch_size"
        self.patch_embed = PatchEmbed1D(in_ch, patch_size, dim)
        num_patches = signal_len // patch_size

        # CLS token (for downstream classification)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = PosEmbed(num_patches + 1, dim)

        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # First we do a patch embedding
        tokens = self.patch_embed(x)              # (B, N, dim)
        # Then we prepend the CLS token
        cls = self.cls.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        # Add positional encoding
        x = self.pos(x)
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # Final normalization
        x = self.norm(x)                          # (B, 1+N, dim)
        # Return CLS token and patch tokens separately
        cls_tok, patch_tok = x[:, 0], x[:, 1:]    # (B, dim), (B, N, dim)
        return cls_tok, patch_tok


# ------------------------------------------------------------------ #
# 5. Lightweight ViT decoder with CAE‑style skip connections
# ------------------------------------------------------------------ #
class ViTDecoderEEG(nn.Module):
    """Fusiona los *patch tokens* con las skip‑connections del DDPM.

    Flujo:
    1. Tokens → blocks → recon_seq (B, out_ch, L)
    2. recon_seq (avg‑pool) se concatena con x0, x_hat, up1 y dd_dn1
    3. Un único bloque de upsampling produce `out` con la misma forma
       que la señal original.
    """

    def __init__(
        self,
        *,
        dim: int,
        out_ch: int,
        patch_size: int,
        n_feat: int = 256,
        depth: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj_in = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, out_ch * patch_size)
        self.pool = nn.AvgPool1d(2)
        # Entradas: pool(x0) + pool(x_hat) + pool(recon_seq) + up1 + dd_dn1
        concat_ch = out_ch * 5  # suponiendo dd_dn1 tiene out_ch
        self.up2 = UnetUp(concat_ch, out_ch, 1, gn=8, factor=2)

    def tokens_to_seq(self, tok: torch.Tensor) -> torch.Tensor:
        patches = self.proj_out(tok)              # (B, N, out_ch*patch)
        return rearrange(patches, "b n (c p) -> b c (n p)", p=self.patch_size)  # (B, out_ch, L)

    def forward(self, x0: torch.Tensor, patch_tok: torch.Tensor, diffusion_out):
        # 1. Transformer path → recon_seq
        x = self.proj_in(patch_tok)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        recon_seq = self.tokens_to_seq(x)         # (B, out_ch, L)

        # 2. DDPM skips
        x_hat, (dd_dn1, dd_dn2), _, _ = diffusion_out # ORIGINAL

        concat = torch.cat([
            self.pool(x0),                # (B, out_ch, L/2)
            self.pool(x_hat.detach()),    # (B, out_ch, L/2)
            self.pool(recon_seq),         # (B, out_ch, L/2)
            dd_dn1.detach()               # (B, out_ch, L/2)  (ajustar si difiere)
        ], dim=1)

        out = self.up2(concat)            # (B, out_ch, L)
        return out                        # salida final para gap loss



# ------------------------------------------------------------------ #
# 7. Complete Diff‑E‑ViT model
# ------------------------------------------------------------------ #
class DiffEViT(nn.Module):
    """Wrapper that mimics the original DiffE forward signature."""

    def __init__(self, encoder: ViTEncoder1D, decoder: ViTDecoderEEG, fc: LinearClassifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc

    def forward(self, x0: torch.Tensor, ddpm_out):
        cls_tok, patch_tok = self.encoder(x0)
        out = self.decoder(x0, patch_tok, ddpm_out)
        fc_out = self.fc(cls_tok)
        return out, fc_out # recon_seq can be used with L1 loss against x0


# ------------------------------------------------------------------ #
# 8. Convenience constructor
# ------------------------------------------------------------------ #

def build_diff_e_vit(
    *,
    in_ch: int = 64,
    signal_len: int = 800,
    patch_size: int = 25,
    dim: int = 256,
    depth: int = 2,
    heads: int = 4,
    latent_dim: int = 512,
    num_classes: int = 5,
    n_feat_decoder: int = 256,
):
    encoder = ViTEncoder1D(
        in_ch=in_ch,
        signal_len=signal_len,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
    )

    decoder = ViTDecoderEEG(
        dim=dim,
        out_ch=in_ch,  # reconstruct original channel count
        patch_size=patch_size,
        n_feat=n_feat_decoder,
    )

    fc = LinearClassifier(dim, latent_dim, num_classes)
    return DiffEViT(encoder, decoder, fc)






























































































































































# # ANTERIOR CODE ----------------------------------------------------------------------------------------------------
# # ==================================================================================================================


# class Encoder(nn.Module):
#     def __init__(self, in_channels, dim=512):
#         super(Encoder, self).__init__()

#         self.in_channels = in_channels
#         # Outputs after two downsampling blocks
#         self.e1_out = dim
#         self.e2_out = dim

#         # Two downsampling layers
#         self.down1 = UnetDown(in_channels, self.e1_out, kernel_size=1, gn=8, factor=2)
#         self.down2 = UnetDown(self.e1_out, self.e2_out, kernel_size=1, gn=8, factor=2)

#         # Global pooling
#         self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)

#     def forward(self, x0):
#         # Downsampling
#         dn1 = self.down1(x0)  # e.g., 800 -> 400
#         dn2 = self.down2(dn1)  # e.g., 400 -> 200
#         # Feature summary
#         z = self.avg_pooling(dn2).view(-1, self.e2_out)
#         return (dn1, dn2), z


# class Decoder(nn.Module):
#     def __init__(self, in_channels, n_feat=256, encoder_dim=512, n_classes=13):
#         super(Decoder, self).__init__()

#         self.in_channels = in_channels
#         self.n_feat = n_feat
#         # Encoder outputs: two levels
#         self.e1_out = encoder_dim
#         self.e2_out = encoder_dim
#         # DDPM outputs: two levels
#         self.d1_out = n_feat
#         self.d2_out = n_feat * 2
#         # Upsampling dims
#         self.u1_out = n_feat

#         # First upsampling: combine second-level encoder and DDPM
#         self.up1 = UnetUp(self.d2_out + self.e2_out, self.u1_out, kernel_size=1, gn=8, factor=2)
#         # Final upsampling & fusion with input & DDPM/encoder first level
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="nearest"),
#             nn.Conv1d(
#                 self.d1_out + self.u1_out + in_channels * 2, in_channels, kernel_size=1
#             ),
#         )
#         self.pool = nn.AvgPool1d(2)

#     def forward(self, x0, encoder_out, diffusion_out):
#         # Encoder features
#         (dn1, dn2), _ = encoder_out
#         # DDPM outputs
#         x_hat, (dd_dn1, dd_dn2), _, _ = diffusion_out

#         # Upsample path
#         up1 = self.up1(torch.cat([dn2, dd_dn2.detach()], dim=1))  # 200 -> 400
#         # Final fusion
#         out = self.up2(
#             torch.cat([
#                 self.pool(x0), self.pool(x_hat.detach()), up1, dd_dn1.detach()
#             ], dim=1)
#         )  # 400 -> 800
#         return out


# class DiffE(nn.Module):
#     def __init__(self, encoder, decoder, fc):
#         super(DiffE, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.fc = fc

#     def forward(self, x0, ddpm_out):
#         encoder_out = self.encoder(x0)
#         decoder_out = self.decoder(x0, encoder_out, ddpm_out)
#         fc_out = self.fc(encoder_out[1])
#         return decoder_out, fc_out




















# # NEW CODE ----------------------------------------------------------------------------------------------------
# # =============================================================================================================

# # vit_autoencoder.py
# import math, torch
# import torch.nn as nn
# from einops import rearrange

# # ------------------------------------------------------------------ #
# # 1. Pequeña utilidad: Positional Encoding aprendible
# # ------------------------------------------------------------------ #
# class PosEmbed(nn.Module):
#     def __init__(self, num_tokens, dim):
#         super().__init__()
#         self.pos = nn.Parameter(torch.randn(1, num_tokens, dim))

#     def forward(self, x):
#         return x + self.pos[:, : x.size(1)]

# # ------------------------------------------------------------------ #
# # 2. Patch embedding (2-D). Sustituye Conv2d por Conv1d para EEG
# # ------------------------------------------------------------------ #
# class PatchEmbed2D(nn.Module):
#     """
#     Conv2d con kernel = patch_size y stride = patch_size.
#     Convierte (B, C, H, W) -> (B, N_patches, dim)
#     """
#     def __init__(self, in_ch, patch_size, dim):
#         super().__init__()
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(
#             in_ch, dim, kernel_size=patch_size, stride=patch_size
#         )

#     def forward(self, x):
#         x = self.proj(x)                            # (B, dim, H/p, W/p)
#         x = rearrange(x, "b d h w -> b (h w) d")    # (B, N, dim)
#         return x

# # ------------------------------------------------------------------ #
# # 3. Bloque Transformer genérico (pre-norm)
# # ------------------------------------------------------------------ #
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, heads, mlp_ratio=4.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn  = nn.MultiheadAttention(
#             dim, heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         hidden = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, dim)
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#         x = x + self.mlp(self.norm2(x))
#         return x

# # ------------------------------------------------------------------ #
# # 4. **Encoder** ViT (stack de bloques)
# # ------------------------------------------------------------------ #
# class ViTEncoder(nn.Module):
#     def __init__(self, *, in_ch, img_size, patch_size,
#                  dim, depth, heads):
#         super().__init__()
#         assert img_size % patch_size == 0, \
#             "img_size debe ser múltiplo de patch_size"
#         self.patch_embed = PatchEmbed2D(in_ch, patch_size, dim)
#         num_patches = (img_size // patch_size) ** 2

#         # token [CLS] para clasificación downstream
#         self.cls = nn.Parameter(torch.randn(1, 1, dim))
#         self.pos = PosEmbed(num_patches + 1, dim)

#         self.blocks = nn.ModuleList([
#             TransformerBlock(dim, heads) for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         x = self.patch_embed(x)                     # (B, N, dim)
#         cls = self.cls.expand(x.size(0), -1, -1)    # (B, 1, dim)
#         x = torch.cat([cls, x], dim=1)              # prepend [CLS]
#         x = self.pos(x)
#         for blk in self.blocks:
#             x = blk(x)
#         return self.norm(x)                         # (B, 1+N, dim)

# # ------------------------------------------------------------------ #
# # 5. **Decoder** ViT (más ligero, opcional)
# # ------------------------------------------------------------------ #
# class ViTDecoder(nn.Module):
#     """
#     Simplificado: toma latentes + mask tokens opcionales y
#     reconstruye patches (p.ej. MAE). Para EEG usarás conv1d transpose.
#     """
#     def __init__(self, *, dim, out_ch,
#                  patch_size, depth=4, heads=8):
#         super().__init__()
#         self.proj_in = nn.Linear(dim, dim)
#         self.pos = None   # puedes reutilizar PosEmbed si reconstruyes en orden

#         self.blocks = nn.ModuleList([
#             TransformerBlock(dim, heads) for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(dim)

#         # Reconstrucción de cada patch → pixeles (patch_size²·out_ch)
#         self.patch_size = patch_size
#         self.proj_out = nn.Linear(dim, out_ch * patch_size * patch_size)

#     def forward(self, tokens):
#         x = self.proj_in(tokens)
#         if self.pos is not None:
#             x = self.pos(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#         # descarta token [CLS] si lo hubiera
#         x = x[:, 1:, :]      # (B, N_patches, dim)
#         x = self.proj_out(x)
#         # Rearrangement depende de tu dominio. Para imágenes:
#         # x -> (B, C, H, W). Para EEG 1-D ajusta reshape.
#         return x

# # ------------------------------------------------------------------ #
# # 6. **Autoencoder completo** (entrenamiento)
# # ------------------------------------------------------------------ #
# class ViTAutoencoder(nn.Module):
#     def __init__(self, encoder: ViTEncoder, decoder: ViTDecoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x, return_latent=False):
#         latent = self.encoder(x)        # (B, 1+N, dim)
#         recon  = self.decoder(latent)   # (B, ...)
#         if return_latent:
#             return recon, latent
#         return recon


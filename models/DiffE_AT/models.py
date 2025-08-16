import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from utils import *


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# Swish activation function
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


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

































class TemporalSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, attn_dim=None, dropout=0.0, window=None):
        super().__init__()
        d_model = attn_dim or channels
        self.proj_in  = nn.Linear(channels, d_model) if d_model != channels else nn.Identity()
        self.mha      = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff       = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, channels) if d_model != channels else nn.Identity()
        self.window   = window  # p.ej. 64 o None para global

    def forward(self, x):  # x: [B, C, T]
        B, C, T = x.shape
        h = x.permute(0, 2, 1)  # [B, T, C]
        h = self.proj_in(h)

        if self.window is None or self.window >= T:
            attn_out, _ = self.mha(h, h, h, need_weights=False)         # global
            h = self.norm1(h + attn_out)
        else:
            W = self.window
            out = torch.empty_like(h)
            for start in range(0, T, W):
                end = min(start+W, T)
                chunk = h[:, start:end, :]
                attn_chunk, _ = self.mha(chunk, chunk, chunk, need_weights=False)
                out[:, start:end, :] = self.norm1(chunk + attn_chunk)
            h = out

        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)
        y = self.proj_out(h).permute(0, 2, 1)  # [B, C, T]
        return y

class ChannelSelfAttention(nn.Module):
    """
    Atención a lo largo del eje canal: tratamos C como "secuencia".
    """
    def __init__(self, channels, num_heads=4, attn_dim=None, dropout=0.0):
        super().__init__()
        d_model = attn_dim or channels
        self.proj_in  = nn.Linear(channels, d_model) if d_model != channels else nn.Identity()
        self.mha      = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff       = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, channels) if d_model != channels else nn.Identity()

    def forward(self, x):  # [B, C, T]
        B, C, T = x.shape
        h = x.permute(0, 2, 1)            # [B, T, C]
        h = self.proj_in(h)               # [B, T, d]
        h = h.transpose(1, 2)             # [B, d, T]
        h = h.permute(0, 2, 1)            # [B, T, d]  (vamos a usar C como "longitud")
        # Reinterpretamos: secuencia = canales; batch = B*T
        h = h.reshape(B*T, 1, -1)         # [B*T, 1, d] -> prepararemos QKV por canal
        # Construimos una secuencia de longitud C: necesitamos [B*T, C, d]
        # Recuperamos las features por canal desde x:
        feats = x.permute(0, 2, 1)        # [B, T, C]
        feats = self.proj_in(feats)       # [B, T, d]
        feats = feats.reshape(B*T, 1, -1) # [B*T, 1, d]  (cada "paso" resumido; simple)
        # Truco simple: usar la misma representación para cada canal no es ideal; alternativa:
        # usar directamente x como [B, T, C] y cambiar a mha con embed por canal.
        # Para mantenerlo simple y estable, hacemos atención tipo SE mejorada:
        # Usamos promedios temporales y aplicamos MHA sobre canales agregados.
        # Implementación compacta y estable:
        c_tokens = x.mean(dim=2)                  # [B, C]
        c_tokens = self.proj_in(c_tokens)         # [B, d]
        c_tokens = c_tokens.unsqueeze(1)          # [B, 1, d]
        # Construimos "consulta" global y "claves/valores" por canal aproximados:
        q = c_tokens
        k = c_tokens
        v = c_tokens
        attn_out, _ = self.mha(q, k, v, need_weights=False)   # [B, 1, d]
        h = self.norm1(c_tokens + attn_out)
        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)
        out = self.proj_out(h).squeeze(1)         # [B, d] -> pesos por canal tras proyección
        # Convertimos a [B, C, 1] con una cabeza tipo SE (simple y eficaz)
        w = torch.sigmoid(nn.Linear(out.size(-1), x.size(1), device=x.device)(out)).unsqueeze(-1)
        return x * w                              # [B, C, T]

class AxialAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4, attn_dim=None, dropout=0.0, window=None):
        super().__init__()
        self.temporal = TemporalSelfAttention(channels, num_heads, attn_dim, dropout, window)
        self.channel  = ChannelSelfAttention(channels, num_heads, attn_dim, dropout)

    def forward(self, x):  # [B, C, T]
        x = self.temporal(x)
        x = self.channel(x)
        return x
































# NUEVO
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResidualConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride=1, gn=8,
                 use_axial_attn=True, heads=4, attn_dim=None, dropout=0.0, win=None):
        super().__init__()
        self.same_channels = inc == outc
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, kernel_size, stride, get_padding(kernel_size)),
            nn.GroupNorm(gn, outc),
            nn.PReLU(),
        )
        self.attn = AxialAttention1D(outc, num_heads=heads, attn_dim=attn_dim, dropout=dropout, window=win) \
                    if use_axial_attn else nn.Identity()
        # (Opcional) mantener SE como modulador final
        self.se = SEBlock(outc)

    def forward(self, x):
        y = self.conv(x)
        y = self.attn(y)   # atención temporal + canal
        y = self.se(y)     # modulación por canal (barato y suele ayudar)
        return (x + y)/2 if self.same_channels else y



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


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        generic one layer FC NN for embedding things  
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.PReLU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


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

class Encoder(nn.Module):
    def __init__(self, in_channels, dim=512):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        # Outputs after two downsampling blocks
        self.e1_out = dim
        self.e2_out = dim

        # Two downsampling layers
        self.down1 = UnetDown(in_channels, self.e1_out, kernel_size=1, gn=8, factor=2)
        self.down2 = UnetDown(self.e1_out, self.e2_out, kernel_size=1, gn=8, factor=2)

        # Global pooling
        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x0):
        # Downsampling
        dn1 = self.down1(x0)  # e.g., 800 -> 400
        dn2 = self.down2(dn1)  # e.g., 400 -> 200
        # Feature summary
        z = self.avg_pooling(dn2).view(-1, self.e2_out)
        return (dn1, dn2), z


class Decoder(nn.Module):
    def __init__(self, in_channels, n_feat=256, encoder_dim=512, n_classes=13):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        # Encoder outputs: two levels
        self.e1_out = encoder_dim
        self.e2_out = encoder_dim
        # DDPM outputs: two levels
        self.d1_out = n_feat
        self.d2_out = n_feat * 2
        # Upsampling dims
        self.u1_out = n_feat

        # First upsampling: combine second-level encoder and DDPM
        self.up1 = UnetUp(self.d2_out + self.e2_out, self.u1_out, kernel_size=1, gn=8, factor=2)
        # Final upsampling & fusion with input & DDPM/encoder first level
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(
                self.d1_out + self.u1_out + in_channels * 2, in_channels, kernel_size=1
            ),
        )
        self.pool = nn.AvgPool1d(2)

    def forward(self, x0, encoder_out, diffusion_out):
        # Encoder features
        (dn1, dn2), _ = encoder_out
        # DDPM outputs
        x_hat, (dd_dn1, dd_dn2), _, _ = diffusion_out

        # Upsample path
        up1 = self.up1(torch.cat([dn2, dd_dn2.detach()], dim=1))  # 200 -> 400
        # Final fusion
        out = self.up2(
            torch.cat([
                self.pool(x0), self.pool(x_hat.detach()), up1, dd_dn1.detach()
            ], dim=1)
        )  # 400 -> 800
        return out


class DecoderNoDiff(nn.Module):
    def __init__(self, in_channels, n_feat=256, encoder_dim=512, n_classes=13):
        super(DecoderNoDiff, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        # Encoder outputs: two levels
        self.e1_out = encoder_dim
        self.e2_out = encoder_dim
        # Upsampling dims
        self.u1_out = n_feat

        # Time and context embeddings (optional)
        self.sin_emb = SinusoidalPosEmb(n_feat)
        self.timeembed1 = EmbedFC(n_feat, self.e2_out)
        self.contextembed1 = EmbedFC(self.e2_out, self.e2_out)

        # Upsampling: from encoder features only
        self.up1 = UnetUp(self.e2_out, self.u1_out, kernel_size=1, gn=8, factor=2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(self.u1_out + self.e1_out + in_channels, in_channels, kernel_size=1),
        )
        self.pool = nn.AvgPool1d(2)

    def forward(self, x0, encoder_out, t):
        # Encoder features
        dn1, dn2 = encoder_out[0]
        z = encoder_out[1]
        # Time embedding
        temb = self.timeembed1(self.sin_emb(t)).view(-1, self.e2_out, 1)
        ctx = self.contextembed1(z).view(-1, self.e2_out, 1)

        # Upsample path
        up1 = self.up1(dn2 + temb + ctx)  # 200 -> 400
        out = self.up2(
            torch.cat([self.pool(x0), up1, dn1], dim=1)
        )  # 400 -> 800
        return out


class DiffE(nn.Module):
    def __init__(self, encoder, decoder, fc):
        super(DiffE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc

    def forward(self, x0, ddpm_out):
        encoder_out = self.encoder(x0)
        decoder_out = self.decoder(x0, encoder_out, ddpm_out)
        fc_out = self.fc(encoder_out[1])
        return decoder_out, fc_out



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

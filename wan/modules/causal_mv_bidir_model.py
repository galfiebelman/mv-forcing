# Adapted from Wan2.1's wan/modules/model.py (https://github.com/Wan-Video/Wan2.1)
# and Self-Forcing's wan/modules/causal_model.py (https://github.com/guandeh17/Self-Forcing,
# commit 33593df), licensed under the Apache License 2.0. Modified for MV-Forcing.
# See NOTICE.md and licenses/LICENSE.Apache-2.0.

"""
CausalWanMVBidirModel: Causal temporal attention + Bidirectional view attention.

Key difference from CausalWanMVModel (causal_mv_model.py):
- Temporal self_attn: CAUSAL (flex_attention with KV cache) — same as CausalWanModel
- View mvs_attn: BIDIRECTIONAL (standard flash attention) — same as WanMVModel
- No view KV cache, no view causal mask

This is the correct architecture for MV-Forcing:
- Temporal causality enables autoregressive rolling for long video
- View bidirectionality enables proper cross-view consistency
"""

from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    WanSelfAttention,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from einops import rearrange
import torch.nn.functional as F


# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

def _rope_time_delta_mul_(k_chunk: torch.Tensor, freqs: torch.Tensor, delta_frames: int) -> None:
    """
    k_chunk: [B, L_sink, H, D] (view of the sink portion of K, with RoPE already applied)
    freqs  : [1024, C/2] complex (self.freqs)
    delta_frames: how many frames to shift the sink to the left (negative) / right (positive)
    In-place, multiplies only the time-axis channels by exp(i * ω * delta_frames)
    """
    if delta_frames == 0:
        return

    B, L, H, D = k_chunk.shape
    assert D % 2 == 0
    c = D // 2
    t_c = c - 2 * (c // 3)   # time channel complex dim
    h_c = c // 3
    w_c = c // 3
    # freqs -> time / height / width split
    freqs_t, _, _ = freqs.split([t_c, h_c, w_c], dim=1)  # [1024, t_c], complex

    #  Complex rotation factor corresponding to the delta (for the time axis)
    shift = abs(int(delta_frames))
    max_pos = freqs_t.shape[0] - 1
    if shift > max_pos:
        shift = max_pos
    mult = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
    mult = mult.view(1, 1, 1, t_c)  # [1,1,1,t_c]

    # Convert only the time-axis channels to complex and multiply (in-place)
    time_ri = k_chunk[..., : 2 * t_c]                                            # [B,L,H,2*t_c]
    time_cx = torch.view_as_complex(time_ri.to(torch.float64).reshape(-1, t_c, 2))  # [(B*L*H), t_c]
    time_cx = time_cx * mult.to(time_cx.dtype)                                   # delta rotate
    time_ri_new = torch.view_as_real(time_cx).reshape(B, L, H, t_c, 2).flatten(-2)  # [B,L,H,2*t_c]
    time_ri.copy_(time_ri_new.to(time_ri.dtype))  # in-place

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with KV cache for temporal dimension."""

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=None, current_start=0, cache_start=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # Training mode (teacher forcing or regular)
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query, roped_key = [], []
                for ii in range(2):
                    roped_query.append(rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v))
                    roped_key.append(rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v))
                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                          device=q.device, dtype=v.dtype)], dim=1)
            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)], dim=1)
            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)], dim=1)

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            # Inference mode with KV cache
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            sink_tokens_v =  self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                num_rolled_tokens_v = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens_v
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                #     kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens_v:sink_tokens_v + num_rolled_tokens_v] = \
                    kv_cache["v"][:, sink_tokens_v + num_evicted_tokens:sink_tokens_v + num_evicted_tokens + num_rolled_tokens_v].clone()
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

            key_win = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            val_win = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Actual sink length (tokens) currently filled in the cache
                sink_len_tokens = min(kv_cache["local_end_index"].item(), sink_tokens)

                # tail (recent) range
                tail_end = local_end_index
                tail_start = max(sink_tokens, local_end_index - self.max_attention_size + sink_tokens)

                # Calculate absolute frames
                tail_len_tokens = tail_end - tail_start
                tail_len_frames = tail_len_tokens // frame_seqlen
                sink_len_frames = sink_len_tokens // frame_seqlen

                tail_start_abs_frame = current_start_frame - tail_len_frames
                desired_sink_abs_start = tail_start_abs_frame - sink_len_frames

                # sink delta rotation (in-place) — only time-axis channels
                if self.sink_size > 0 and sink_len_tokens > 0:
                    if "sink_base_abs_start_frame" not in kv_cache:
                        kv_cache["sink_base_abs_start_frame"] = torch.tensor(desired_sink_abs_start,
                                                                             device=kv_cache["k"].device)
                    delta = int(desired_sink_abs_start - kv_cache["sink_base_abs_start_frame"].item())
                    if delta != 0:
                        _rope_time_delta_mul_(kv_cache["k"][:, :sink_len_tokens], freqs, delta)
                        kv_cache["sink_base_abs_start_frame"].fill_(desired_sink_abs_start)

                    key_win = torch.cat([
                        kv_cache["k"][:, :sink_len_tokens],
                        kv_cache["k"][:, tail_start:tail_end]
                    ], dim=1)
                    val_win = torch.cat([
                        kv_cache["v"][:, :sink_len_tokens],
                        kv_cache["v"][:, tail_start:tail_end]
                    ], dim=1)
                else:
                    key_win = kv_cache["k"][:, tail_start:tail_end]
                    val_win = kv_cache["v"][:, tail_start:tail_end]

            x = attention(roped_query, key_win, val_win)
            # x = attention(
            #     roped_query,
            #     kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
            #     kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            # )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanMVBidirAttentionBlock(nn.Module):
    """
    Attention block with:
    - Causal temporal self-attention (flex_attention + KV cache)
    - Bidirectional cross-view attention (standard flash attention, NO view cache)
    - Cross-attention to text
    - FFN
    """

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads,
                 local_attn_size=-1, sink_size=0, qk_norm=True,
                 cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        # Temporal self-attention (CAUSAL with KV cache)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)

        # Cross-view attention (BIDIRECTIONAL, standard flash attention)
        self.norm_mvs = WanLayerNorm(dim, eps)
        self.mvs_attn = WanSelfAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.cam_encoder = nn.Linear(12, dim)
        self.projector = nn.Linear(dim, dim)
        self.modulation_mvs = nn.Parameter(torch.randn(1, 3, dim) / dim ** 0.5)

        # Cross-attention to text
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)

        # FFN
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # Temporal modulation (6 components)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        self._init_syncam_components()

    def _init_syncam_components(self):
        """Initialize SynCamMaster-specific components with zero init."""
        self.cam_encoder.weight.data.zero_()
        self.cam_encoder.bias.data.zero_()
        self.projector.weight.data.zero_()
        self.projector.bias.data.zero_()
        self.modulation_mvs.data = self.modulation.data[:, :3, :].clone()
        # self.mvs_attn.load_state_dict(self.self_attn.state_dict(), strict=True)

    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
        cam_emb, freqs_mvs, block_mask,
        kv_cache=None, crossattn_cache=None, current_start=0, cache_start=None,
    ):
        """
        Args:
            x: [B, L, C] where B = num_views (2 during training)
            e: [B, F, 6, C] timestep embeddings
            cam_emb: [V, F, 12] camera embeddings
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        # Temporal modulation
        first_e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # 1) Causal temporal self-attention (with KV cache during inference)
        y = self.self_attn(
            (self.norm1(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + first_e[1]) + first_e[0]).flatten(1, 2),
            seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start
        )
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * first_e[2]).flatten(1, 2)

        # 2) Bidirectional cross-view attention (NO cache, standard flash attention)
        shift_mvs, scale_mvs, gate_mvs = (self.modulation_mvs.unsqueeze(1) + e[:, :, :3,:]).chunk(3, dim=2)
        input_x = (self.norm_mvs(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + scale_mvs) + shift_mvs).flatten(1, 2)

        v, f, _ = cam_emb.shape
        h, w = 30, 52  # hardcoded for Wan 1.3B at 480p
        B_v = x.shape[0]  # batch_size * num_views
        b = B_v // v
        mvs_batch_size = b * f
        mvs_seq_len = torch.tensor([v * h * w], device=x.device, dtype=torch.long).repeat(mvs_batch_size)
        mvs_grid_sizes = torch.tensor([[v, h, w]], device=x.device, dtype=torch.long).repeat(mvs_batch_size, 1)

        cam_emb_enc = self.cam_encoder(cam_emb)
        cam_emb_enc = cam_emb_enc.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        cam_emb_enc = rearrange(cam_emb_enc, 'b f h w d -> b (f h w) d')

        input_x = input_x + cam_emb_enc
        input_x = rearrange(input_x, '(b v) (f h w) d -> (b f) (v h w) d', v=v, f=f, h=h, w=w)
        x = rearrange(x, '(b v) (f h w) d -> (b f) (v h w) d', v=v, f=f, h=h, w=w)

        # Bidirectional attention over views — no block_mask, no kv_cache
        att = self.mvs_attn(input_x, mvs_seq_len, mvs_grid_sizes, freqs_mvs)
        proj = self.projector(att)
        x = x + (proj.unflatten(dim=1, sizes=(v, frame_seqlen)) * gate_mvs.permute(1,0,2,3)).flatten(1, 2)  # proj_reshaped
        x = rearrange(x, '(b f) (v h w) d -> (b v) (f h w) d', v=v, f=f, h=h, w=w)

        # 3) Cross-attention to text + FFN
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
            x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, first_e, crossattn_cache)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x


class CausalWanMVBidirModel(ModelMixin, ConfigMixin):
    """
    Wan DiT with causal temporal attention + bidirectional view attention.
    Used as the student generator in MV-Forcing.
    """

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['CausalWanMVBidirAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 local_attn_size=-1,
                 sink_size=0,
                 use_cut3r=False):
        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_cut3r = use_cut3r
        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        if use_cut3r:
            self.rendered_embedding = nn.Conv3d(
                in_dim + 1, dim, kernel_size=patch_size, stride=patch_size)
            nn.init.zeros_(self.rendered_embedding.weight)
            nn.init.zeros_(self.rendered_embedding.bias)

        # Blocks with causal temporal + bidirectional view attention
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanMVBidirAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # Head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.init_weights()
        self.gradient_checkpointing = False
        self.block_mask = None
        self.num_frame_per_block = 3
        self.independent_first_frame = False

    def init_cut3r(self):
        self.rendered_embedding = nn.Conv3d(
            self.in_dim + 1, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        nn.init.zeros_(self.rendered_embedding.weight)
        nn.init.zeros_(self.rendered_embedding.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device, num_frames=21, frame_seqlen=1560, num_frame_per_block=1, local_attn_size=-1
    ):
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        frame_indices = torch.arange(0, total_length, frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (
                    kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length,
            _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"  cached blockwise causal mask: block_size={num_frame_per_block} frames")

        return block_mask

    def _forward_inference(
        self, x, t, context, seq_len,
        cam_emb=None, clip_fea=None, y=None,
        kv_cache=None, crossattn_cache=None,
        current_start=0, cache_start=0, z_rendered=None,
    ):
        """
        Inference with KV cache. Processes one temporal block at a time.
        x is a list of tensors, one per batch element (batch=2 for [GT_view0, noisy_view1]).
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        if z_rendered is not None:
            z_r = [self.rendered_embedding(u.unsqueeze(0)) for u in z_rendered]
            x = [xi + zi for xi, zi in zip(x, z_r)]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
            freqs=self.freqs, context=context, context_lens=context_lens,
            cam_emb=cam_emb, freqs_mvs=self.freqs,
            block_mask=self.block_mask,
        )
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    block, x, **kwargs, use_reentrant=False)
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self, x, t, context, seq_len,
        cam_emb=None, clean_x=None, aug_t=None,
        clip_fea=None, y=None, z_rendered=None,
    ):
        """Full-sequence training forward (no KV cache)."""
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.block_mask is None:
            if clean_x is not None:
                raise NotImplementedError("Teacher forcing not implemented for MV bidir model")
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device, num_frames=x.shape[2],
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        if z_rendered is not None:
            z_r = [self.rendered_embedding(u.unsqueeze(0)) for u in z_rendered]
            x = [xi + zi for xi, zi in zip(x, z_r)]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
            freqs=self.freqs, context=context, context_lens=context_lens,
            cam_emb=cam_emb, freqs_mvs=self.freqs,
            block_mask=self.block_mask,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)
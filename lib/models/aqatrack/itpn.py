# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import math
import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple
from lib.models.aqatrack.base_backbone import BaseBackbone
from lib.utils.utils import combine_tokens, recover_tokens

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, init_values=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            if self.attn is not None:
                self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            else:
                self.gamma_1 = None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rpe_index=None, mask=None):
        if self.gamma_2 is None:
            if self.attn is not None:
                x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.attn is not None:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rpe_index, mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = self.patch_shape = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        in_chans = 6
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


# the spatial size are split into 4 patches (then downsample 2x)
# concat them and then reduce them to be of 2x channels.
#
class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# PatchSplit is for upsample
# used in FPN for downstream tasks such as detection/segmentation
class PatchSplit(nn.Module):
    def __init__(self, dim, fpn_dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(
            B, N, H, W, 2, 2, self.fpn_dim
        ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, N, 2 * H, 2 * W, self.fpn_dim
        )
        return x


class iTPN(BaseBackbone):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, fpn_dim=256, fpn_depth=2,
                 embed_dim=256, mlp_depth1=3, mlp_depth2=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, init_values=None,
                 norm_layer=nn.LayerNorm, ape=True, rpe=False, patch_norm=True, use_checkpoint=False,
                 num_outs=3, **kwargs):
        super().__init__()
        assert num_outs in [1, 2, 3, 4, 5]
        self.num_classes = num_classes
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_outs = num_outs
        self.num_main_blocks = depth
        self.fpn_dim = fpn_dim
        self.mlp_depth1 = mlp_depth1
        self.mlp_depth2 = mlp_depth2
        self.depth = depth

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            # Hp_z, Hp_z
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, mlp_depth1 + mlp_depth2 + depth))
        self.blocks = nn.ModuleList()

        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(mlp_depth1)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(mlp_depth2)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer, init_values=init_values
            ) for _ in range(depth)]
        )

        self.norm_ = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.absolute_pos_embed
        patch_pos_embed = self.absolute_pos_embed
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def get_num_layers(self):
        return self.mlp_depth1 + self.mlp_depth2 + self.depth

    def forward_features(self, z,x,soft_token_template_mask):
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]

        _, _, H_z, W_z = z.shape
        Hp_z, Wp_z = H_z // self.patch_embed.patch_size[0], W_z // self.patch_embed.patch_size[1]

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        for blk in self.blocks[:-self.num_main_blocks]:
            x =  blk(x)
            z = blk(z)

        x = x[..., 0, 0, :]
        z = z[..., 0, 0, :]

        x += self.pos_embed_x
        z += self.pos_embed_z

        token_type_search = self.token_type_search.expand(B, x.shape[1], -1)
        token_type_template_bg = self.token_type_template_bg.expand(B, z.shape[1], -1)
        token_type_template_fg = self.token_type_template_fg.expand(B, z.shape[1], -1)

        x += token_type_search

        token_type_template_fg = soft_token_template_mask*token_type_template_fg
        token_type_template_bg = (1- soft_token_template_mask) * token_type_template_bg
        z += token_type_template_fg + token_type_template_bg

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            # cls_tokens = cls_tokens + self.cls_pos_embed
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = blk(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        x = self.norm_(x)

        return x,aux_dict


    def forward(self, z,x,soft_token_template_mask, **kwargs):
        x,aux_dict = self.forward_features(z,x,soft_token_template_mask)
        return x,aux_dict


@register_model
def itpn_base_3324_patch16_224(pretrained="", **kwargs):
    model = iTPN(
        patch_size=16, embed_dim=512, mlp_depth1=3, mlp_depth2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            pretrained, map_location="cpu"
        )
        ###### naive method
        # missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        #### process patch_embed.proj
        normal_weights = {}
        # special_weights = {}
        for k, v in checkpoint.items():
            if "patch_embed.proj.weight" in k:
                normal_weights[k] = torch.cat([v, v], dim=1)
            else:
                normal_weights[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(normal_weights, strict=False)

    return model


@register_model
def itpn_large_2240_patch16_224(pretrained=False, **kwargs):
    model = iTPN(
        patch_size=16, embed_dim=768, mlp_depth1=2, mlp_depth2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.aqatrack.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = True
        self.add_sep_seg = False

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = False

        patch_pos_embed_all = self.pos_embed
        patch_pos_embed = patch_pos_embed_all[:,1:,:]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size // self.patch_size, self.img_size // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(torch.cat([template_patch_pos_embed,template_patch_pos_embed],dim=1))
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        cls_token = patch_pos_embed_all[:,0,:].unsqueeze(1)
        self.cls_token = nn.Parameter(cls_token)

        # token_type
        self.token_type_search = nn.Parameter(torch.zeros(1, 1, 512))
        self.token_type_template_fg = nn.Parameter(torch.zeros(1, 1, 512))
        self.token_type_template_bg = nn.Parameter(torch.zeros(1, 1, 512))


        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, mask=None):
        B = x.shape[0]
    
        z = self.patch_embed(z)
        x = self.patch_embed(x)
    
        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)
            z = blk(z)

        x = x[..., 0, 0, :]
        z = z[..., 0, 0, :]


        z += self.pos_embed_z
        x += self.pos_embed_x

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        
        x = combine_tokens(z, x, mode=self.cat_mode)
        #x = combine_tokens(x, z,  mode=self.cat_mode)


        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            # cls_tokens = cls_tokens + self.cls_pos_embed
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = blk(x)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        x = self.norm_(x)

        return x, aux_dict

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic HiViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x,)

        return x, aux_dict

import copy
from tkinter import N
from typing import Optional, List
import warnings
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from functools import partial
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from .position_encoding import build_memory_position_encoding, build_query_position_encoding

from lib.models.transformers.position_encoding import PositionEmbeddingNone,PositionEmbeddingLearned


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)


    def forward(self, query, key, value=None,
                attn_mask=None, key_padding_mask=None,
                need_weights=False):
        """
        query: [B, N, C]
        attn_mask: [N, N] torch.float32
        key_padding_mask: [B, N] torch.bool
        """
        B, q_N, C = query.shape
        
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(1), key.size(1)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [B * self.num_heads, query.size(1), key.size(1)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = self.q(query).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,head,HW,C/head
        k = self.k(key).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, head, N, N)
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask
        
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)

        if need_weights:
            # average attention weights over heads
            return x, attn.sum(dim=1) / self.num_heads  # attn: (B, q_len, k_len)
        else:
            return x, None


class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): # nn.ReLU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=0, mlp_ratio=1, 
                dropout=0., vocab_size=1001, memory_pos_embedding=None, query_pos_embedding=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.query_embedding = nn.Embedding(vocab_size, self.d_model)  # (1001, 256)
        self.memory_pos_embedding = memory_pos_embedding
        self.query_pos_embedding = query_pos_embedding

        decoder_layer = TransformerDecoderLayer(d_model=d_model, num_heads=nhead, mlp_ratio=mlp_ratio,
                                                attn_drop=dropout, proj_drop=dropout)
        if num_decoder_layers == 0:
            self.layers = None
        else:
            self.layers = _get_clones(decoder_layer, num_decoder_layers)

        self.num_decoder_layers = num_decoder_layers
        self.scale_factor = float(d_model // nhead) ** 0.5

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def memory_mask_pos_enc(self, attn_mask, feat_sz):
        """
        attn_mask: (B, img_H, img_W)
        feat_sz: feature size
        """
        batch_size = attn_mask.size(0)
        attn_mask = attn_mask.to(torch.float32)
        attn_mask = F.interpolate(attn_mask.unsqueeze(1), size=(feat_sz, feat_sz)).to(torch.bool).squeeze(1)

        pos_embeds = self.memory_pos_embedding(attn_mask) # sine position encoding  (B, C, feat_sz, feat_sz)

        attn_mask = attn_mask.view(batch_size, -1)
        pos_embeds = pos_embeds.view(batch_size, self.d_model, -1).transpose(1, 2)

        return attn_mask, pos_embeds


    def tri_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).float().transpose(0, 1)
        mask.masked_fill_(mask == 0, float('-inf'))
        mask.masked_fill_(mask == 1, float(0.))
        return mask


    def forward(self, query, memory, memory_key_padding_mask, memory_pos,
                return_intermediate_output=False, is_inference=False):
        """
        query:        (B, 5, C) or (B, 18, C) or (B, bbox+mask, C)
        memory:       (B, HW, C)
        memory_key_padding_mask:  (B, HW)
        memory_pos:   (B, HW, C)
        """
        # learnable pos encoding
        query_pos = self.query_pos_embedding(query)

        query_mask = self.tri_mask(query.size(1)).to(query.device)

        aux_logits = []
        output = query
        for i in range(len(self.layers)):
            output, attn_weights = self.layers[i](output, memory,
                                                query_pos=query_pos, memory_pos=memory_pos,
                                                query_mask=query_mask,
                                                memory_key_padding_mask=memory_key_padding_mask,
                                                need_weights=False)
            if return_intermediate_output and i < len(self.layers)-1:
                aux_logits.append(output)
        
        if is_inference:
            return output
        else:
            return output, aux_logits


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8,dim_feedforward=1024, mlp_ratio=8,
                 dropout=0.1, proj_drop=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # self.activation = F.relu()
        self.normalize_before = False


    def forward(self, query, memory,
                query_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                query_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = False):
        """
        query: (B, 5, C)
        memory: (B, HW, C)
        query_pos: (B, 5, C)
        memory_pos: (B, HW, C)
        query_mask: (5,5)
        memory_key_padding_mask: (B, HW)
        """
        # query self attn
        # q = k = with_pos_embed(query, query_pos)
        # query2, selfattn_weights = self.self_attn(q, k, value=query, attn_mask=query_mask, need_weights=need_weights)
        # query = query + self.dropout1(query2)
        # query = self.norm1(query)
        #
        # memory self attn
        q = k = with_pos_embed(memory, memory_pos)
        query2, selfattn_weights = self.self_attn(q, k, value=memory, attn_mask=memory_key_padding_mask, need_weights=need_weights)
        memory = memory + self.dropout1(query2)
        memory = self.norm1(memory)

        # cross attention
        query2, crossattn_weights = self.multihead_attn(query=with_pos_embed(query, query_pos),
                                                    key=with_pos_embed(memory, memory_pos),
                                                    value=memory,
                                                    key_padding_mask=memory_key_padding_mask,
                                                    need_weights=need_weights)
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # mlp
        query2 = self.linear2(self.dropout(F.relu(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        return query

class TransformerDecoder_v0(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=0, mlp_ratio=1,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, mlp_ratio=mlp_ratio,
                                                dropout=dropout, proj_drop=dropout)
        if num_decoder_layers == 0:
            self.layers = None
        else:
            self.layers = _get_clones(decoder_layer, num_decoder_layers)

        self.num_decoder_layers = num_decoder_layers
        self.scale_factor = float(d_model // nhead) ** 0.5

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def memory_mask_pos_enc(self, attn_mask, feat_sz):
        """
        attn_mask: (B, img_H, img_W)
        feat_sz: feature size
        """
        batch_size = attn_mask.size(0)
        attn_mask = attn_mask.to(torch.float32)
        attn_mask = F.interpolate(attn_mask.unsqueeze(1), size=(feat_sz, feat_sz)).to(torch.bool).squeeze(1)

        pos_embeds = self.memory_pos_embedding(attn_mask)  # sine position encoding  (B, C, feat_sz, feat_sz)

        attn_mask = attn_mask.view(batch_size, -1)
        pos_embeds = pos_embeds.view(batch_size, self.d_model, -1).transpose(1, 2)

        return attn_mask, pos_embeds


    def forward(self, query, memory, query_pos, memory_pos):
        """
        query:        (B, 5, C) or (B, 18, C) or (B, bbox+mask, C)
        memory:       (B, HW, C)
        memory_key_padding_mask:  (B, HW)
        memory_pos:   (B, HW, C)
        """
        # learnable pos encoding
        # query_pos = self.query_pos_embedding(query)

        output = query
        for i in range(len(self.layers)):
            output = self.layers[i](output, memory,
                                                  query_pos=query_pos, memory_pos=memory_pos,
                                                  memory_key_padding_mask=None,
                                                  need_weights=False)


        return output

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., num_vlfusion_layers=0,
                vl_input_type='separate'):
        super().__init__()
        # self.multihead_attn = MultiheadAttention(dim, num_heads=num_heads, 
        #                                     qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.vl_input_type = vl_input_type
        VLFusion_layer = MultiheadAttention(dim, num_heads=num_heads, 
                                            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        if num_vlfusion_layers == 0:
            self.VLFusion_layers = None
        else:
            self.VLFusion_layers = _get_clones(VLFusion_layer, num_vlfusion_layers)

    def forward(self, query, memory,
                query_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                query_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = False):

        if self.vl_input_type == 'separate':
            output = query
        elif self.vl_input_type == 'concat':
            output, query_pos = torch.cat([memory, query], dim=1), torch.cat([memory_pos, query_pos], dim=1)
            memory, memory_pos = output.clone(), query_pos.clone()
            memory_key_padding_mask = torch.cat([memory_key_padding_mask, query_key_padding_mask], dim=1)

        for layer in self.VLFusion_layers:
            output, attn_weights = layer(query=with_pos_embed(output, query_pos),
                                        key=with_pos_embed(memory, memory_pos),
                                        value=memory,
                                        key_padding_mask=memory_key_padding_mask,
                                        need_weights=need_weights)      # attn_weights: (B, q_len, k_len)
            
            output = query * output

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_decoder(cfg):
    memory_pos_embedding = build_memory_position_encoding(cfg)
    query_pos_embedding = build_query_position_encoding(cfg)
    return TransformerDecoder(
        d_model=cfg.MODEL.DECODER.HIDDEN_DIM,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=cfg.MODEL.DECODER.NUM_HEADS,
        mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
        num_decoder_layers=cfg.MODEL.DECODER.DEC_LAYERS,
        vocab_size=cfg.MODEL.DECODER.VOCAB_SIZE,
        memory_pos_embedding=memory_pos_embedding,
        query_pos_embedding=query_pos_embedding,
    )

from lib.models.transformers.position_encoding import PositionEmbeddingNone,PositionEmbeddingLearned
def build_temporal_decoder(cfg):
    memory_pos_embedding = PositionEmbeddingLearned(15,256)
    query_pos_embedding = PositionEmbeddingLearned(15,256)
    return TransformerDecoder_v0(
        d_model=cfg.MODEL.DECODER.HIDDEN_DIM,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=cfg.MODEL.DECODER.NUM_HEADS,
        mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
        num_decoder_layers=2,
        vocab_size=cfg.MODEL.DECODER.VOCAB_SIZE,
        memory_pos_embedding=memory_pos_embedding,
        query_pos_embedding=query_pos_embedding,
    )

def build_text_prompt_decoder(cfg):
    memory_pos_embedding = PositionEmbeddingLearned(15,512)
    query_pos_embedding = PositionEmbeddingLearned(15,512)
    return TransformerDecoder_v0(
        d_model=512,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=cfg.MODEL.DECODER.NUM_HEADS,
        mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
        num_decoder_layers=2,
        vocab_size=cfg.MODEL.DECODER.VOCAB_SIZE,
        memory_pos_embedding=memory_pos_embedding,
        query_pos_embedding=query_pos_embedding,
    )

def build_img_prompt_decoder(cfg):
    memory_pos_embedding = PositionEmbeddingLearned(15,256)
    query_pos_embedding = PositionEmbeddingLearned(15,256)
    return TransformerDecoder_v0(
        d_model=cfg.MODEL.DECODER.HIDDEN_DIM,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=8,
        mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
        num_decoder_layers=2,
        vocab_size=cfg.MODEL.DECODER.VOCAB_SIZE,
        memory_pos_embedding=memory_pos_embedding,
        query_pos_embedding=query_pos_embedding,
    )


def build_temporal_fusion(cfg):
    return TransformerDecoder_v0(
        d_model=512,
        dropout=0.1,
        nhead=8,
        mlp_ratio=1,
        num_decoder_layers=1  #3
    )


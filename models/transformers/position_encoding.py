"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from lib.utils.misc import NestedTensor


class PositionEmbeddingSine1D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors # [B, L, C]
        mask = tensor_list.mask # [B, L]
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, L]
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # [B, L, C]
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # [B, L, C]
        # pos = pos_x.permute(0, 2, 1)    # [B, C, L]
        return pos_x


class PositionEmbeddingSine2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        # mask: (B, H, W)
        not_mask = ~mask  # (batch,h,w)  ~mask:按位取反
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # cumulative sum along axis 1 (h axis) --> (b, h, w)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # cumulative sum along axis 2 (w axis) --> (b, h, w)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 2π * (y / sigma(y))  (16,8,8)
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 2π * (x / sigma(x))  (16,8,8)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)  # (0,1,2,...,d/2)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (b,h,w,d/2)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2) (16,8,8,128)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (b,h,w,d/2) (16,8,8,128)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (b,h,w,d) (16,8,8,256) to (16,256,8,8)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embed = nn.Embedding(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x):
        # x: (B, 5, C) or (B, mask, C) or (B, bbox+mask, C)
        n = x.size(1)
        i = torch.arange(n, device=x.device)
        pos = self.embed(i).unsqueeze(0).repeat(x.size(0), 1, 1) # (N,C) --> (1,N,C) --> (B,N,C)
        return pos


class PositionEmbeddingNone(nn.Module):
    """
    No positional encoding.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.n_dim = num_pos_feats * 2

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        b, _, h, w = x.size()
        return torch.zeros((b, self.n_dim, h, w), device=x.device)  # (B, C, H, W)


def build_memory_position_encoding(cfg):
    N_steps = cfg.MODEL.DECODER.HIDDEN_DIM // 2  # N_steps = 128
    if cfg.MODEL.DECODER.MEMORY_POSITION_EMBEDDING in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine2D(N_steps, normalize=True)
    elif cfg.MODEL.DECODER.MEMORY_POSITION_EMBEDDING in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif cfg.MODEL.DECODER.MEMORY_POSITION_EMBEDDING in ('None', ):
        print("Not using positional encoding.")
        position_embedding = PositionEmbeddingNone(N_steps)
    else:
        raise ValueError(f"not supported {cfg.MODEL.DECODER.MEMORY_POSITION_EMBEDDING}")

    return position_embedding


def build_query_position_encoding(cfg):
    bbox_task = cfg.TRAIN.BBOX_TASK
    language_task = getattr(cfg.TRAIN, "LANGUAGE_TASK", False)
        
    if bbox_task and language_task:      # bbox and language tasks
        seq_in_dim = (4+1) + (1+1)
    elif bbox_task and not language_task:  # bbox level
        seq_in_dim = 4+1

    N_steps = cfg.MODEL.DECODER.HIDDEN_DIM
    if cfg.MODEL.DECODER.QUERY_POSITION_EMBEDDING in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine2D(N_steps, normalize=True)
    elif cfg.MODEL.DECODER.QUERY_POSITION_EMBEDDING in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(in_dim=seq_in_dim, out_dim=N_steps)
    elif cfg.MODEL.DECODER.QUERY_POSITION_EMBEDDING in ('None', ):
        print("Not using positional encoding.")
        position_embedding = PositionEmbeddingNone(N_steps)
    else:
        raise ValueError(f"not supported {cfg.MODEL.DECODER.QUERY_POSITION_EMBEDDING}")

    return position_embedding
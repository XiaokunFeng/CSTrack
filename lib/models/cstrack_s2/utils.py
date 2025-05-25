import math

import torch
import torch.nn.functional as F

def combine_tokens(template_tokens, search_tokens, mode='direct', return_res=False):
    # [B, HW, C]
    len_t = template_tokens.shape[1]
    len_s = search_tokens.shape[1]

    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
    elif mode == 'template_central':
        central_pivot = len_s // 2
        first_half = search_tokens[:, :central_pivot, :]
        second_half = search_tokens[:, central_pivot:, :]
        merged_feature = torch.cat((first_half, template_tokens, second_half), dim=1)
    elif mode == 'partition':
        feat_size_s = int(math.sqrt(len_s))
        feat_size_t = int(math.sqrt(len_t))
        window_size = math.ceil(feat_size_t / 2.)
        # pad feature maps to multiples of window size
        B, _, C = template_tokens.shape
        H = W = feat_size_t
        template_tokens = template_tokens.view(B, H, W, C)
        pad_l = pad_b = pad_r = 0
        # pad_r = (window_size - W % window_size) % window_size
        pad_t = (window_size - H % window_size) % window_size
        template_tokens = F.pad(template_tokens, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, Hp // window_size, window_size, W, C)
        template_tokens = torch.cat([template_tokens[:, 0, ...], template_tokens[:, 1, ...]], dim=2)
        _, Hc, Wc, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, -1, C)
        merged_feature = torch.cat([template_tokens, search_tokens], dim=1)

        # calculate new h and w, which may be useful for SwinT or others
        merged_h, merged_w = feat_size_s + Hc, feat_size_s
        if return_res:
            return merged_feature, merged_h, merged_w

    else:
        raise NotImplementedError

    return merged_feature

'''
add token transfer to feature
'''
def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x


'''
feature2token
'''
def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens
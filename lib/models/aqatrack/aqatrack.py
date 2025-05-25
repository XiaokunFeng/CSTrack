"""
Basic AQATrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.aqatrack.hivit import hivit_small, hivit_base
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.layers.transformer_dec import build_transformer_dec
from lib.models.layers.position_encoding import build_position_encoding
from lib.utils.misc import NestedTensor


import queue

class AQATrack(nn.Module):
    """ This is the base class for AQATrack """

    def __init__(self, transformer, box_head, transformer_dec, position_encoding, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.transformer_dec = transformer_dec
        self.position_encoding = position_encoding 
        self.query_embed=nn.Embedding(num_embeddings=1, embedding_dim=512)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                return_last_attn=False,
                training=True, #True
                tgt_pre = None,
                ):
        b0, num_search = template[0].shape[0], len(search)
        if training:
            search = torch.cat(search, dim=0)
            template = template[0].repeat(num_search,1,1,1)

        x, aux_dict = self.backbone(z=template, x=search,
                                    return_last_attn=return_last_attn, ) #x=[B,N,C]
        
        b,n,c = x.shape
        input_dec = x
        batches = [[] for _ in range(b0)]
        for i, input in enumerate(input_dec):
            batches[i % b0].append(input.unsqueeze(0))
        x_decs = []
        query_embed = self.query_embed.weight
        assert len(query_embed.size()) in [2, 3]
        if len(query_embed.size()) == 2:
            query_embeding = query_embed.unsqueeze(1)
        for i,batch in enumerate(batches):
            if len(batch) ==0:
                continue
            tgt_all = [torch.zeros_like(query_embeding) for _ in range(num_search)]

            for j, input in enumerate(batch):
                pos_embed = self.position_encoding(1)
                tgt_q = tgt_all[j]
                tgt_kv = torch.cat(tgt_all[:j+1], dim=0)
                if not training and len(tgt_pre) != 0:
                    tgt_kv = torch.cat(tgt_pre, dim=0)
                tgt = [tgt_q, tgt_kv]
                tgt_out = self.transformer_dec(input.transpose(0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
                x_decs.append(tgt_out[0])
                tgt_all[j] = tgt_out[0]#
            if not training:
                if len(tgt_pre) < 3: #num_search-1 
                    tgt_pre.append(tgt_out[0])
                else:
                    tgt_pre.pop(0)
                    tgt_pre.append(tgt_out[0])
            
        batch0 =[]
        if not training:
            batch0.append(x_decs[0])
        else:
            batch0 = [x_decs[i + j*num_search] for j in range(b0) for i in range(num_search)]
        
        x_dec = torch.cat(batch0, dim = 1)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, x_dec, None) # STM and head

        out.update(aux_dict)
        out['tgt'] = tgt_pre 
        #print(len(out['tgt']), 'out[tgt]')
        return out

    def forward_head(self, cat_feature, out_dec=None, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # STM
        enc_opt = cat_feature[:, -self.feat_len_s:] 
        dec_opt = out_dec.transpose(0,1).transpose(1,2) 
        att = torch.matmul(enc_opt, dec_opt)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        #Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_aqatrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('AQATrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
        backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    
    transformer_dec = build_transformer_dec(cfg, hidden_dim)
    position_encoding = build_position_encoding(cfg, sz = 1)

    box_head = build_box_head(cfg, hidden_dim)
    model = AQATrack(
        backbone,
        box_head,
        transformer_dec,
        position_encoding,        
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'AQATrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

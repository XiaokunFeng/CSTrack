"""
CSTrack Model
"""
import os

import torch
from torch import nn

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
### aqatrack
from lib.models.aqatrack.hivit import hivit_small, hivit_base
from lib.models.aqatrack.fast_itpn import fast_itpn_base_3324_patch16_224

from lib.models.transformers.transformer import build_rgb_det_decoder

from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.transformers.transformer_temporal import build_temporal_fusion


import torch.nn.functional as F
from lib.models.layers.frozen_bn import FrozenBatchNorm2d

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
class ConfidencePred(nn.Module):
    def __init__(self):
        super(ConfidencePred, self).__init__()
        self.feat_sz = 24
        self.stride = 1
        self.img_sz = self.feat_sz * self.stride
        freeze_bn = False

        self.conv1_ctr = conv(5, 16, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(16, 16 // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(16 // 2, 16 // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(16 // 4, 16 // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(16 // 8, 1, kernel_size=1)

        self.fc1 = nn.Linear(256, 512)

        self.fc2 = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x,xz_feature=None, gt_score_map=None):
        """ Forward pass with input x. """

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # flatten
        x = score_map_ctr.flatten(1)
        x = self.relu(self.fc1(x))

        x = self.sigmoid(self.fc2(x))

        return x

class TaskClsBranch(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=4):
        super(TaskClsBranch, self).__init__()
        # 3层 MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



class CSTRACK_S2(nn.Module):
    def __init__(self, transformer,  box_head, aux_loss=False, head_type="CORNER",cfg=None):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.backbone = transformer
        # self.backbone_det = backbone_det
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        ## involve task classification
        self.task_cls = TaskClsBranch()

        ## rgb det fusion
        self.rgb_prompts = nn.Embedding(num_embeddings=4, embedding_dim=512)
        self.det_prompts = nn.Embedding(num_embeddings=4, embedding_dim=512)

        self.query_len = 1
        self.cls_prompts_pos = nn.Embedding(num_embeddings=self.query_len+8, embedding_dim=512)  # pos for cur query

        self.backbone_rgb_det_fusion = build_rgb_det_decoder(cfg)
        self.confidence_pred = ConfidencePred()

        ### temporal modeling
        self.temporal_len = 4
        self.dy_template_pos_embed = nn.Embedding(num_embeddings=self.temporal_len,
                                                  embedding_dim=512)  # pos for cur query
        self.temporal_fusion = build_temporal_fusion(cfg)


    def forward_backbone(self, template, search, cls_token,soft_token_template_mask,x_pos):
        # template b, 12, h,w
        # search b,6,h,w
        template_rgb = [template[:, :3, :, :], template[:, 6:9, :, :]]
        search_rgb = search[:, :3, :, :]

        template_det = [template[:, 3:6, :, :], template[:, 9:12, :, :]]
        search_det = search[:, 3:, :, :]

        soft_token_template_mask = [soft_token_template_mask[:, :64], soft_token_template_mask[:, 64:]]

        x_rgb, token_type_infor = self.backbone.forward_features_pe(z=template_rgb, x=search_rgb, soft_token_template_mask =soft_token_template_mask)
        x_det,_ = self.backbone.forward_features_pe(z=template_det, x=search_det, soft_token_template_mask = soft_token_template_mask)

        ## rgb dte fusion
        rgb_prompts = self.rgb_prompts.weight.unsqueeze(0).repeat(x_det.shape[0],1,1)
        det_prompts = self.det_prompts.weight.unsqueeze(0).repeat(x_det.shape[0],1,1)
        x_rgb = torch.cat([rgb_prompts,x_rgb],dim=1)
        x_det = torch.cat([det_prompts,x_det],dim=1)

        search_pos = x_pos[:,self.query_len+4:]
        x = self.backbone_rgb_det_fusion(x_rgb, x_det, search_pos, search_pos)

        x, aux_dict = self.backbone.forward_features_stage3(x, cls_token,x_pos)
        return x, aux_dict

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                soft_token_template_mask=None,
                update_cls_token=None,
                temporal_infor=[],
                first_frame_flag=False,
                training=True):

        b0, num_search = template[0].shape[0], len(search)
        if training:
            search = torch.cat(search, dim=0)
            template_rgb_with_det = torch.cat(template, dim=1)  # (bs,12(rgb0;det0;rgb1;det1),w,h)
            soft_token_template_mask_rgb_with_det = torch.cat(soft_token_template_mask,
                                                              dim=1)  # (bs,128(mask0;mask1),1)

            template_rgb_with_det_temporal = []
            soft_token_template_mask_rgb_with_det_temporal = []
            for _ in range(num_search):
                template_rgb_with_det_temporal.append(template_rgb_with_det)
                soft_token_template_mask_rgb_with_det_temporal.append(soft_token_template_mask_rgb_with_det)
            template_rgb_with_det_temporal = torch.cat(template_rgb_with_det_temporal, dim=0)
            soft_token_template_mask_rgb_with_det_temporal = torch.cat(soft_token_template_mask_rgb_with_det_temporal,
                                                                       dim=0)

        else:
            b0 = 1
            template_rgb_with_det_temporal = torch.cat(template, dim=1)
            soft_token_template_mask_rgb_with_det_temporal = torch.cat(soft_token_template_mask, dim=1)


        # x, aux_dict = self.backbone(z=template, x=search,
        #                             soft_token_template_mask = soft_token_template_mask )
        cls_prompts_pos = self.cls_prompts_pos.weight.unsqueeze(0)
        x_pos_0 = torch.cat([cls_prompts_pos, self.backbone.pos_embed_z, self.backbone.pos_embed_x], dim=1)
        # pos_embed = x_pos.transpose(0, 1).repeat(1, b0, 1)
        x_pos = x_pos_0.repeat(b0*num_search, 1, 1)
        x, aux_dict = self.forward_backbone(template_rgb_with_det_temporal, search, update_cls_token, soft_token_template_mask_rgb_with_det_temporal,
                                                 x_pos)

        #### Forward task cls
        cls_token = x[:, 0, :]
        task_res = self.task_cls(cls_token)
        #### forward temporal information

        pre_temporal_pos = self.dy_template_pos_embed.weight.unsqueeze(1).repeat(b0, 1, 16)
        pre_temporal_pos = pre_temporal_pos.view(b0, 4 * 16, 512)
        x_pos = x_pos_0.repeat(b0, 1, 1)
        output_list = []

        for temporal_index in range(num_search):
            x_item = x[temporal_index * b0:(temporal_index + 1) * b0]

            x_f = x_item[:, -256:]
            x_f1 = torch.matmul(x_f, x_f.permute(0, 2, 1))
            x_f = torch.matmul(x_f1, x_f)

            z_f = x_item[:, :-256]

            x_z = torch.matmul(x_f, z_f.permute(0, 2, 1))
            att_map = x_z.mean(-1)

            tensor_min = torch.min(att_map)
            tensor_max = torch.max(att_map)
            # normalized_tensor = (s_vl_1 - tensor_min) / (tensor_max - tensor_min)
            normalized_tensor = (tensor_max - att_map) / (tensor_max - tensor_min)

            attn_xz = normalized_tensor.view(-1, 256)

            ### initialize memory
            if training:
                if temporal_index == 0:
                    z0_values, z0_indices = torch.topk(attn_xz, 16, dim=1)
                    z0_indices = torch.sort(z0_indices, dim=1).values
                    z0_filter = torch.gather(x_item[:, -256:], 1,
                                             z0_indices.unsqueeze(2).expand(-1, -1, x_item.size(-1))).squeeze(2)
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(z0_filter)
            else:
                if first_frame_flag:
                    z0_values, z0_indices = torch.topk(attn_xz, 16, dim=1)
                    z0_indices = torch.sort(z0_indices, dim=1).values
                    z0_filter = torch.gather(x_item[:, -256:], 1,
                                             z0_indices.unsqueeze(2).expand(-1, -1, x_item.size(-1))).squeeze(2)
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(z0_filter)

            #### cross_attention with temporal_infor
            temporal_infor_data = torch.cat(temporal_infor, dim=1)
            xt_f = self.temporal_fusion(x_item.permute((1, 0, 2)), temporal_infor_data.permute((1, 0, 2)),
                                        x_pos.permute((1, 0, 2)),
                                        pre_temporal_pos.permute((1, 0, 2)))
            xt_f = xt_f.permute(1, 0, 2)

            # STM
            enc_opt = xt_f[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

            # get the tracking result & confidence pred
            out_item = self.forward_head(opt_feat)  # STM and head

            #### heatmap by gaussion of pred_bbox
            pred_boxes_item = out_item["pred_boxes"]
            gaussion_map = self.gen_gaussion_heatmap(pred_boxes_item)
            attn_xz = attn_xz + gaussion_map
            att_map_debug = attn_xz[0].view(16, 16).detach().cpu().numpy()
            z0_values, z0_indices = torch.topk(attn_xz, 16, dim=1)
            z0_indices = torch.sort(z0_indices, dim=1).values

            z0_filter = torch.gather(x_item[:, -256:], 1, z0_indices.unsqueeze(2).expand(-1, -1, x_item.size(-1))).squeeze(2)
            if training:
                if temporal_index == 0:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(z0_filter)
                else:
                    temporal_infor[1:3] = temporal_infor[2:]
                    temporal_infor[-1] = z0_filter
            else:
                if first_frame_flag:
                    temporal_infor = []
                    for _ in range(self.temporal_len):
                        temporal_infor.append(z0_filter)
                else:
                    # temporal_infor_debug = temporal_infor.copy()
                    temporal_infor[1:3] = temporal_infor[2:]
                    temporal_infor[-1] = z0_filter

            out_item["attn_xz"] = attn_xz
            out_item["gaussion_map"] = gaussion_map
            output_list.append(out_item)

        ### final output
        out = {}
        keys_list = list(output_list[0].keys())
        for key_item in keys_list:
            value_item = []
            for out_item in output_list:
                value_item.append(out_item[key_item])

            # value_item_reshape = [value_item[i + j * num_search] for i in range(num_search) for j in range(b0)]
            value_item_reshape = torch.cat(value_item, dim=0)
            out[key_item] = value_item_reshape

        out["task_res"] = task_res
        if training == False:
            out["z0_filter"] = z0_filter
            out["temporal_infor"] = temporal_infor
        return out

    def forward_head(self, opt_feat, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """

        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        Nq = 1
        bs = opt_feat.shape[0]
        # Head
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

            score_map = torch.cat([score_map_ctr, size_map, offset_map], dim=1)
            confidence_pred = self.confidence_pred(score_map)

            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   "confidence_pred": confidence_pred}
            return out
        else:
            raise NotImplementedError

    def gen_gaussion_heatmap(self, pred_bbox, map_size=16):
        # 参数设置
        pred_bbox = pred_bbox * map_size  # cxcywh
        gaussian_2d_res = []
        for batch_index in range(pred_bbox.shape[0]):
            pred_bbox_item = pred_bbox[batch_index]
            center_x, center_y = pred_bbox_item[0, 1], pred_bbox_item[0, 0]
            std_x, std_y = pred_bbox_item[0, 3], pred_bbox_item[0, 2]
            size_x, size_y = map_size, map_size


            x = torch.arange(0, size_x).float()
            y = torch.arange(0, size_y).float()
            x_grid, y_grid = torch.meshgrid(x, y)
            x_grid, y_grid = x_grid.to(pred_bbox.device), y_grid.to(pred_bbox.device)

            # gen gaussian_2d
            gaussian_2d = torch.exp(-0.5 * (((x_grid - center_x) / std_x) ** 2 + ((y_grid - center_y) / std_y) ** 2))
            # gaussian_2d_debug = gaussian_2d.detach().cpu().numpy()
            gaussian_2d_res.append(gaussian_2d.view(1, -1))
        gaussian_2d_res = torch.cat(gaussian_2d_res, dim=0)
        return gaussian_2d_res


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_cstrack_s2(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../resource/pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE  and training and ("CSTRACK" not in cfg.MODEL.PRETRAIN_FILE) :
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'hivit_base_adaptor':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'itpn_base':  # by this
        backbone, backbone_det = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1


    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = CSTRACK_S2(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        cfg=cfg
    )

    if cfg.MODEL.PRETRAINED_PATH != "" and ("CSTRACK" in cfg.MODEL.PRETRAINED_PATH) and training:
        checkpoint = torch.load(os.path.join(cfg.MODEL.PRETRAINED_PATH), map_location="cpu")
        ckpt = checkpoint["net"]
        model_weight = {}
        for k, v in ckpt.items():
            model_weight[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(model_weight, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAINED_PATH)

    return model


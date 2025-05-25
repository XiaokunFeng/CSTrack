from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate,generate_bbox_mask
from lib.train.admin import multigpu
import torch.nn as nn
from lib.utils.misc import NestedTensor


class CSTrackS2Actor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.task_cls_loss_fn = nn.CrossEntropyLoss()
        # reg loss
        self.confidence_reg_loss = nn.MSELoss()

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # assert len(data['template_images']) == 1
        template_list, search_list = [], []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        # search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)
        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img_i)



        # soft token type infor
        bbox_mask_list = []
        for template_item in data["template_anno"]:
            template_bbox = template_item * template_list[0].shape[2]
            bbox_mask = torch.zeros((template_list[0].shape[0], template_list[0].shape[2], template_list[0].shape[3] )).to(template_list[0].device)
            bbox_mask = generate_bbox_mask(bbox_mask, template_bbox )

            bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16)
            bbox_mask = bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0],-1).unsqueeze(-1)
            bbox_mask_list.append(bbox_mask)

        out_dict = self.net(template=template_list,
                            search=search_list,
                            soft_token_template_mask = bbox_mask_list)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        # gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_bbox = gt_dict['search_anno'].view(-1, 4)
        gts = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(gts, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # task cls loss
        task_res = pred_dict['task_res']
        targets = []
        dataset_infor = gt_dict["dataset"]
        for item in dataset_infor:
            if item in ["lasot","lasot_lang","got10k","tnl2k","coco","trackingnet","vasttrack_test"]:
                targets.append(0)
            elif item in ["lasher"]:
                targets.append(1)
            elif item in ["visevent"]:
                targets.append(2)
            elif item in ["depthtrack"]:
                targets.append(3)
            else:
                print("unexpected_ds", item)
        targets = torch.tensor(targets).to(task_res.device)
        targets = targets.repeat(self.settings.num_search)
        task_cls_loss = self.task_cls_loss_fn(task_res,targets)

        ## involve confidence_pred_score
        confidence_pred = pred_dict["confidence_pred"].squeeze(1)
        confidence_loss = self.confidence_reg_loss(confidence_pred.float(), iou.float())

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss  + task_cls_loss + confidence_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/task_cls_loss": task_cls_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
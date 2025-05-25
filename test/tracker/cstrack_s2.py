from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.cstrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.utils.misc import NestedTensor
from lib.models.cstrack_s2 import build_cstrack_s2
from lib.test.tracker.cstrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np
from lib.test.utils.hann import hann2d
from lib.utils.ce_utils import generate_mask_cond,generate_bbox_mask
from matplotlib import pyplot as plt
# from pytorch_pretrained_bert import BertTokenizer

def get_resize_template_bbox(template_bbox, resize_factor ):
    w,h = template_bbox[2] , template_bbox[3]
    w_1, h_1 = int(w * resize_factor )  , int( h*resize_factor )
    xc, yc = 64, 64

    x0,y0 = int( xc - w_1*0.5 ) , int( yc - h_1*0.5 )

    resize_template_bbox = [x0,y0,w_1,h_1]

    return  resize_template_bbox

def visualize_grid_attention_v2(img, attention_mask, ratio=1, cmap="jet", save_image=True,
                                save_path="./test.jpg", quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    # print("load image from: ", img_path)
    # img = Image.open(img_path, mode='r')
    img_h, img_w = 256, 256
    plt.clf()
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    # img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    # img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 256).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=quality)
    #



class CSTRACKS2(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CSTRACKS2, self).__init__(params)
        network = build_cstrack_s2(params.cfg,training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print("load from ",self.params.checkpoint)

        self.cfg = params.cfg
        self.seq_format = self.cfg.DATA.SEQ_FORMAT
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = params.debug
        self.frame_id = 0
        self.temporal_infor = []
        self.first_frame_flag = True

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        if DATASET_NAME in ["LASHER", "RGBT234"]:
            self.update_threshold = 0.45  # 0.45
        else:
            self.update_threshold = 0.7
        print("Update threshold is: ", self.update_threshold)

        #multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT
        print("MULTI_MODAL_VISION is: ", self.multi_modal_vision)

        # task cls
        self.task_cls_res = None

    def initialize(self, image, info: dict):
        # if (self.multi_modal_vision == True) and (image.shape[-1] == 3):
        #     image = np.concatenate((image, image), axis=-1)

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)

        self.template_list = [template] * self.num_template


        # soft token type infor
        template_bbox = info['init_bbox']  # xywh
        resize_template_bbox = get_resize_template_bbox(template_bbox, resize_factor)

        resize_template_bbox = [torch.tensor(resize_template_bbox).to(template.device)]
        bbox_mask = torch.zeros((1, self.params.template_size, self.params.template_size))
        bbox_mask = generate_bbox_mask(bbox_mask, resize_template_bbox)

        bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16)
        bbox_mask = bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0], -1).unsqueeze(-1)

        bbox_mask = bbox_mask.to(template.device)

        self.soft_token_template_mask = [bbox_mask,bbox_mask]


        # get the initial sequence i.e., [start]
        batch = template.shape[0]

        self.state = info['init_bbox']
        self.frame_id = 0

        self.temporal_infor = []
        self.first_frame_flag = True

    def track(self, image, info: dict = None):
        # if (self.multi_modal_vision == True) and (image.shape[-1] == 3):
        #     image = np.concatenate((image, image), axis=-1)

        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)

        # search_list = [search]

        # run the encoder
        with torch.no_grad():
            out_dict = self.network(self.template_list, search, self.soft_token_template_mask,
                                    temporal_infor = self.temporal_infor,
                                    first_frame_flag = self.first_frame_flag,training=False)

        self.first_frame_flag = False
        self.temporal_infor = out_dict["temporal_infor"]

        max_value, max_index = torch.max(out_dict["task_res"], dim=1)
        self.task_cls_res = max_index[0].cpu().tolist()


        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'],
                                                                out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        conf_score = max_score  # the confidence score
        # if max_score < 0.475:
        #     print("confidence score: ", conf_score, "in ",self.frame_id)
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                               output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and (template.size(1) == 3):
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                # soft token type infor
                template_bbox = self.state  # xywh
                resize_template_bbox = get_resize_template_bbox(template_bbox, resize_factor)

                resize_template_bbox = [torch.tensor(resize_template_bbox).to(template.device)]
                bbox_mask = torch.zeros((1, self.params.template_size, self.params.template_size))
                bbox_mask = generate_bbox_mask(bbox_mask, resize_template_bbox)

                bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16)
                bbox_mask = bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0], -1).unsqueeze(-1)

                bbox_mask = bbox_mask.to(template.device)

                self.soft_token_template_mask.append(bbox_mask)
                if len(self.soft_token_template_mask) > self.num_template:
                    self.soft_token_template_mask.pop(1)


        return {"target_bbox": self.state,
                "best_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)




def get_tracker_class():
    return CSTRACKS2

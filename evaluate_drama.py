import argparse
import multiprocessing as mp
import os
import time
import cv2
import sys
import json
import torch
from torchvision.ops import box_iou

import logging
from tqdm import tqdm
import numpy as np
import copy

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
# sys.path.insert(0, '/data1/yubo/risk/GRiT/')
sys.path.insert(1, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config
from grit.predictor import VisualizationDemo

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# constants
WINDOW_NAME = "GRiT"

class Eval_Drama():
    def __init__(self, predict_json_name) -> None:
        self.root = '/data1/yubo/risk/GRiT/'

        self.predict_json_name = predict_json_name
        self.predict_json_path = os.path.join(self.root, 'eval/predict_json', self.predict_json_name)
        print("create image used {}.".format(self.predict_json_name))

        # logging
        # log_path = open(os.path.join(self.root, 'eval', self.predict_json_name.split('.')[0]+'.log'), encoding="utf-8", mode="w")
        log_path = open('/data1/yubo/risk/GRiT/eval/20231231_1728_drama_b_densecap_model_final.log', encoding="utf-8", mode="w")
        logging.basicConfig(stream=log_path, datefmt = '%d-%m-%Y %H%M%S', 
                            format = '%(asctime)s %(name)s: %(levelname)s: %(message)s',
                            level = logging.INFO)
        logging.info('Start...{}'.format(log_path.name))

        # test image path
        self.img_root = os.path.join(self.root, 'datasets/coco/test2017/')
        # eval/draw
        self.draw = os.path.join(self.root, 'eval/draw', self.predict_json_name.split('.')[0])
        if not os.path.exists(self.draw):
            os.mkdir(self.draw)
        # color
        self._color = [
            {"color": (220, 20, 60)},
            {"color": (119, 11, 32)},
            {"color": (0, 0, 142)},
            {"color": (0, 0, 230)},
            {"color": (106, 0, 228)},
            {"color": (0, 60, 100)},
            {"color": (0, 80, 100)},
            {"color": (0, 0, 70)},
            {"color": (0, 0, 192)},
            {"color": (250, 170, 30)},
            {"color": (100, 170, 30)},
            {"color": (220, 220, 0)},
            {"color": (175, 116, 175)},
            {"color": (250, 0, 30)},
            {"color": (165, 42, 42)},
            {"color": (255, 77, 255)},
            {"color": (0, 226, 252)},
            {"color": (182, 182, 255)},
            {"color": (0, 82, 0)},
            {"color": (120, 166, 157)},      
            {"color": (110, 76, 0)},
            {"color": (174, 57, 255)},
            {"color": (199, 100, 0)},
            {"color": (72, 0, 118)},
            {"color": (255, 179, 240)},
            {"color": (0, 125, 92)},
            {"color": (209, 0, 151)},
            {"color": (188, 208, 182)},
            {"color": (0, 220, 176)},
            {"color": (255, 99, 164)},
            {"color": (92, 0, 73)},
            {"color": (133, 129, 255)},
            {"color": (78, 180, 255)},
            {"color": (0, 228, 0)},
            {"color": (174, 255, 243)},
            {"color": (45, 89, 255)},
            {"color": (134, 134, 103)},
            {"color": (145, 148, 174)},
            {"color": (255, 208, 186)},
            {"color": (197, 226, 255)},   
            {"color": (171, 134, 1)},
            {"color": (109, 63, 54)},
            {"color": (207, 138, 255)},
            {"color": (151, 0, 95)},
            {"color": (9, 80, 61)},
        ]

    def calculate(self, data, count, calculate):
        count['total'] += 1
        logging.info("{},len pred_bbox:{}".format(data["img_name"], len(data["pred_bbox"]) ))
        if(len(data["pred_bbox"])==0):
            count['bbox_no'] += 1
        elif(len(data["pred_bbox"])>1):
            count['bbox_multi'] += 1
        elif(len(data["pred_bbox"])==1):
            count['bbox_only1'] += 1

            img_bbox = data['gt_bbox']
            img_bbox[2], img_bbox[3] = img_bbox[0]+img_bbox[2], img_bbox[1]+img_bbox[3]
            pred_bbox = data['pred_bbox'][0]
            ious = box_iou(torch.tensor(pred_bbox).reshape(-1,4), torch.tensor(img_bbox).reshape(-1,4))
            calculate['mean_iou'].append(ious)
            if(ious>=0.5):
                calculate['count_iou'] += 1         

    def draw_img(self, data): 
        img_name = data["img_name"]
        gt_bbox = data["gt_bbox"]
        pred_bbox = data["pred_bbox"]
        gt_caption = data["gt_captions"]
        pred_caption = data["pred_captions"]
        pred_score = data["scores"]

        img = cv2.imread(os.path.join(self.img_root, img_name))
        img_h, img_w, _ = img.shape
        img_add = np.full((600, img_w, 3), 100, dtype=np.uint8)
        img = np.concatenate((img, img_add), axis=0)
        x_gt, y_gt, w_gt, h_gt = [int(num) for num in gt_bbox]

        base_x = 20
        base_y = img_h + 30  

        # gt 
        cv2.rectangle(img, (x_gt, y_gt), (x_gt+w_gt, y_gt+h_gt), (0, 255, 0), 1)  
        cv2.putText(img, 'gt_caption:', (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(img, gt_caption, (base_x+200, base_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        # predict 
        for i in range(len(pred_bbox)): 
            x2, y2, x3, y3 = [int(num) for num in pred_bbox[i]] 
            cv2.putText(img, 'pre_caption:', (base_x, base_y+40*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, self._color[i]["color"], 1)
            # 画bbox
            cv2.rectangle(img, (x2, y2), (x3, y3), self._color[i]["color"], 2) 
            # 画caption  
            cv2.putText(img, pred_caption[i], (base_x+200, base_y+40*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, self._color[i]["color"], 1) 
            # 画score
            cv2.putText(img, str(round(pred_score[i], 3)), (x2, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._color[i]["color"], 1) 
            
        cv2.imwrite(os.path.join(self.draw, img_name), img) 

    def out_put(self, count, calculate):
        logging.info(f"total={count['total']}")
        logging.info(f"len(data['pred_bbox'])>1:{count['bbox_multi']}")
        logging.info(f"len(data['pred_bbox'])=1:{count['bbox_only1']}")
        logging.info(f"len(data['pred_bbox'])=0:{count['bbox_no']}")

        logging.info(f"total={count['total']}")
        logging.info(f"count_iou={calculate['count_iou']}")
        logging.info(f"mean_iou={sum(calculate['mean_iou'])/count['total']}")
        logging.info(f"acc={calculate['count_iou']/count['total']}")

        logging.info(f"end...")

        print(f"total={count['total']}")
        print(f"len(data['pred_bbox'])>1:{count['bbox_multi']}")
        print(f"len(data['pred_bbox'])=1:{count['bbox_only1']}")
        print(f"len(data['pred_bbox'])=0:{count['bbox_no']}")

        print(f"total={count['total']}")
        print(f"count_iou={calculate['count_iou']}")
        print(f"mean_iou={sum(calculate['mean_iou'])/count['total']}")
        print(f"acc={calculate['count_iou']/count['total']}")

        print(f"end...")

    def main(self):
        # count
        count = {
            'total': 0,
            'bbox_multi': 0,
            'bbox_only1':0,
            'bbox_no':0,
            'caption_multi': 0,
            'caption_only1':0,
            'caption_no':0
        }
        calculate = {
            'count_iou':0,
            'mean_iou':[]
        }

        predict_datas = json.load(open(self.predict_json_path, 'r'))
        # count caption/bbox=0、bbox>1、bbox=1
        for data in tqdm(predict_datas):
            # data1 = copy.deepcopy(data)
            # self.calculate(data1, count, calculate)
            data2 = copy.deepcopy(data)
            self.draw_img(data2)
        # self.out_put(count, calculate)
 

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/GRiT_B_DenseCap_drama.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--input",
        nargs="+",
        default='datasets/coco/test2017',
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='visualization/drama-20231203-2048',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.01,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='DenseCap',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # default=['MODEL.WEIGHTS', 'output/GPU-4/20231231_1728_drama_b_densecap/model_final.pth'],
        # default=['MODEL.WEIGHTS', 'output/GPU-4/20240101_1804_drama_b_densecap/model_final.pth'],
        default=['MODEL.WEIGHTS', 'output/GPU-4/20240101_1200_drama_b_densecap/model_final.pth'],
        # default=['MODEL.WEIGHTS', 'output/GPU-4/20240101_2343_drama_b_densecap/model_final.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_predict_json():
    # json文件路径
    drama_instances_path = 'datasets/coco/annotations/instances_test2017.json'
    drama_caption_path = 'datasets/coco/annotations/captions_test2017.json'
    # instances_test2017.json中"images"和"annotations"字段 
    with open(drama_instances_path, 'r') as instance_file:
        instance_lists = json.load(instance_file)
        json_img_dict = instance_lists["images"]
        json_annotations_dict = instance_lists["annotations"]
    instance_file.close()
    # captions_test2017.json中"annotations"字段 
    with open(drama_caption_path, 'r') as caption_file:
        caption_lists = json.load(caption_file)
        caption_annotations_dict = caption_lists["annotations"]
    caption_file.close()
    # 获取instances_test2017.json中file_name，按照相对位置存放
    fileName_list = []
    for data in json_img_dict:
        fileName_list.append(data['file_name'])

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # 存放处理结果
    json_data = []
    # args.input ：'datasets/coco/test2017' 
    print(f"加载预测数据...") 
    # 获取预测数据, key:file_name; value:predict_data
    predict_data = {}
    if args.input:
        for path in tqdm(os.listdir(args.input)):
            img = read_image(os.path.join(args.input, path), format="BGR")
            predictions, visualized_output = demo.run_on_image(img)
            predict_data[path] = predictions

    # 获取在instances_test2017.json中对应索引
    for key, value in predict_data.items():
        idx = fileName_list.index(key)
        predict_data[key]['idx'] = idx

    # 解析预测数据 
    print(f"解析预测数据...") 
    for file_name, data in tqdm(predict_data.items()):
        gt_bbox = json_annotations_dict[data['idx']]['bbox']
        gt_caption = caption_annotations_dict[data['idx']]['caption']
        load_data = {
            "img_name" : file_name,
            "gt_bbox" : gt_bbox,
            "gt_captions" : gt_caption,
            "pred_bbox" : data['instances'].pred_boxes.tensor.cpu().tolist(),
            "pred_captions" : data['instances'].pred_object_descriptions.data,
            "pred_classes" : data['instances'].pred_classes.cpu().tolist(),
            "logits" : data['instances'].logits.cpu().tolist(),
            "scores" : data['instances'].scores.cpu().tolist()
        }
        json_data.append(load_data)

    # 把预测结果个gt写入json
    save_json_name = args.opts[1].split('/')   
    save_json_name = save_json_name[2] + '_' + save_json_name[3]
    save_json_name = save_json_name.replace('.pth', '_TH0-' + str(args.confidence_threshold).split('.')[1] + '.json')
    save_json_path = os.path.join('eval/predict_json/', save_json_name)
    with open(save_json_path, 'w')as f:
        json.dump(json_data, f, indent=4)
    f.close()


if __name__ == "__main__":
    # get_predict_json()

    # predict_json_name = '20231231_1728_drama_b_densecap_model_final_TH0-5.json'
    # predict_json_name = '20231231_1728_drama_b_densecap_model_final_TH0-2.json'

    # predict_json_name = '20240101_1804_drama_b_densecap_model_final_TH0-5.json'
    # predict_json_name = '20240101_1804_drama_b_densecap_model_final_TH0-2.json'
    # predict_json_name = '20240101_2343_drama_b_densecap_model_final_TH0-2.json'

    # predict_json_name = '20240101_1200_drama_b_densecap_model_final_TH0-2.json'
    # predict_json_name = '20240101_1200_drama_b_densecap_model_final_TH0-1.json'
    predict_json_name = '20240101_1200_drama_b_densecap_model_final_TH0-01.json'

    eval = Eval_Drama(predict_json_name)
    eval.main()








    # data = json.load(open('eval/predict_json/20231231_1728_drama_b_densecap_model_final.json', 'r'))

    # idx = 0

    # img_name = data[idx]["img_name"]
    # gt_bbox = data[idx]["gt_bbox"] 
    # gt_caption = data[idx]["gt_captions"]
    # pred_bbox = data[idx]["pred_bbox"]
    # pred_caption = data[idx]["pred_captions"][0]

    # x_gt, y_gt, w_gt, h_gt = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    # x_pre1, y_pre1, x_pre2, y_pre2 = pred_bbox[0][0], pred_bbox[0][1], pred_bbox[0][2], pred_bbox[0][3]

    # img_show = cv2.imread(os.path.join('datasets/coco/test2017/', img_name))
    # img = img_show.copy()
    # img_h, img_w, _ = img.shape
    # base_x = 30
    # base_y = 30  

    # _color = [
    #     {"color": (0, 0, 255)},
    #     {"color": (244, 35, 232)},
    #     {"color": (70, 70, 70)},
    #     {"color": (102, 102, 156)},
    #     {"color": (190, 153, 153)},
    #     {"color": (250, 170, 30)},
    #     {"color": (220, 220, 0)},
    #     {"color": (107, 142, 35)}
    # ]

    # # gt 
    # cv2.rectangle(img, (x_gt, y_gt), (x_gt+w_gt, y_gt+h_gt), (0, 255, 0), 1)  
    # cv2.putText(img, 'gt_caption:', (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # cv2.putText(img, gt_caption, (base_x+240, base_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    # # pred
    # cv2.putText(img, 'pre_caption:', (base_x, base_y+60), cv2.FONT_HERSHEY_SIMPLEX, 1, _color[0]["color"], 1)
    # x2, y2, x3, y3 = [int(num) for num in pred_bbox[0]] 
    # cv2.rectangle(img, (x2, y2), (x3, y3), _color[0]["color"], 2)   
    # cv2.putText(img, pred_caption, (base_x+240, base_y+60), cv2.FONT_HERSHEY_SIMPLEX, 1, _color[0]["color"], 1) 
    # cv2.imwrite('eval/draw/test_' + str(idx) + '.png', img)

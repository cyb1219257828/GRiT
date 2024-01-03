'''
计算iou
'''

import json, os, cv2, logging
from torchvision.ops import box_iou
import torch
from tqdm import tqdm

def calculate_iou(json_name):
    json_file = '/data1/yubo/risk/GRiT/eval/'
    log_path = '/data1/yubo/risk/GRiT/eval/calculate_bbox/'

    log_name = json_name
    log_name = log_name.split('.')[0] + '.log'
    # 配置日志模块 
    # 日志 
    log_path = open(os.path.join(log_path, log_name), encoding="utf-8", mode="w")
    logging.basicConfig(
        stream=log_path,
        datefmt = '%d-%m-%Y %H%M%S',
        format = '%(asctime)s %(name)s: %(levelname)s: %(message)s',
        level = logging.INFO
    )
    logging.info('开始执行{}'.format(log_path))

    with open(os.path.join(json_file, 'predict_json', json_name), 'r') as f:
        datas = json.load(f)
    f.close()

    total = 0
    count = 0
    no_bbox = 0
    mean_iou = []
    for data in tqdm(datas):
        total += 1
        img_bbox = data['gt_bbox']
        img_bbox[2], img_bbox[3] = img_bbox[0]+img_bbox[2], img_bbox[1]+img_bbox[3]
        if(len(data["pred_bbox"])==0):
            no_bbox += 1
            continue
        elif(len(data["pred_bbox"])>1):
            pred_bbox = data['pred_bbox'][0]
            ious = box_iou(torch.tensor(pred_bbox).reshape(-1,4), torch.tensor(img_bbox).reshape(-1,4))
            mean_iou.append(ious)
            if(ious>=0.5):
                count += 1
        elif(len(data["pred_bbox"])==1):
            pred_bbox = data['pred_bbox'][0]
            ious = box_iou(torch.tensor(pred_bbox).reshape(-1,4), torch.tensor(img_bbox).reshape(-1,4))
            mean_iou.append(ious)
            if(ious>=0.5):
                count += 1            

    logging.info(f"total={total}")
    logging.info(f"count={count}")
    logging.info(f"nobbox={no_bbox}")
    logging.info(f"mean_iou={sum(mean_iou)/total}")
    logging.info(f"acc={count/total}")
    logging.info('结束{}'.format(log_path))
    log_path.close()


if __name__ == '__main__':

    # json_name = '20231231_1728_drama_b_densecap_model_final_TH0-5.json'
    # json_name = '20231231_1728_drama_b_densecap_model_final_TH0-2.json'
    # json_name = '20240101_1804_drama_b_densecap_model_final_TH0-5.json'
    # json_name = '20240101_1804_drama_b_densecap_model_final_TH0-2.json'
    # json_name = '20240101_1200_drama_b_densecap_model_final_TH0-2.json'
    # json_name = '20240101_1200_drama_b_densecap_model_final_TH0-1.json'
    json_name = '20240101_1200_drama_b_densecap_model_final_TH0-01.json'
    # json_name = '20240101_2343_drama_b_densecap_model_final_TH0-2.json'

    calculate_iou(json_name)

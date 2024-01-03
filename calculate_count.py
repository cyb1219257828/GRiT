import json, os, logging

'''
    统计 json 文件 
    bbox>1 的数量
    无bbox的数量
    caption数量
    无caption数量
'''



def calculate_count(json_name):
    json_root = '/data1/yubo/risk/GRiT/eval/predict_json/'
    log_root = '/data1/yubo/risk/GRiT/eval/count'

    log_name = json_name
    log_name = log_name.split('.')[0] + '.log'
    # 配置日志模块 
    # 日志 
    log_path = open(os.path.join(log_root, log_name), encoding="utf-8", mode="w")
    logging.basicConfig(
        stream=log_path,
        datefmt = '%d-%m-%Y %H%M%S',
        format = '%(asctime)s %(name)s: %(levelname)s: %(message)s',
        level = logging.INFO
    )
    logging.info('开始执行{}'.format(log_path))

    with open(os.path.join(json_root, json_name), 'r') as f:
        datas = json.load(f)
    f.close()

    total = 0
    bbox_multi, caption_multi = 0, 0
    no_bbox, no_caption = 0, 0
    bbox_only1, caption_only1 = 0, 0

    for data in datas:
        total += 1

        logging.info(data["img_name"])

        if(len(data["pred_bbox"])>1):
            bbox_multi += 1
            logging.info("pred_bbox:{}".format( len(data["pred_bbox"]) ))
        elif(len(data["pred_bbox"])==1):
            bbox_only1 += 1
            logging.info("pred_bbox:{}".format( len(data["pred_bbox"]) ))
        elif(len(data["pred_bbox"])==0):
            no_bbox += 1
            logging.info("no_bbox:{}".format( len(data["pred_bbox"]) ))

        if(len(data['pred_captions'])>1):
            caption_multi += 1
            logging.info("pred_caption:{}".format( len(data['pred_captions']) ))
        elif(len(data['pred_captions'])==1):
            caption_only1 += 1
            logging.info("pred_caption:{}".format( len(data['pred_captions']) ))
        elif(len(data['pred_captions'])==0):
            no_caption += 1
            logging.info("no_caption:{}".format( len(data['pred_captions']) ))

    logging.info(f"end...\n")
    logging.info(f"total={total}")
    logging.info(f"len(data['pred_bbox'])>1:{bbox_multi}")
    logging.info(f"len(data['pred_bbox'])=1:{bbox_only1}")
    logging.info(f"no_bbox={no_bbox}")
    logging.info(f"len(data['pred_caption'])>1:{caption_multi}")
    logging.info(f"len(data['pred_caption'])=1:{caption_only1}")
    logging.info(f"no_bbox={no_caption}")
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
    calculate_count(json_name)

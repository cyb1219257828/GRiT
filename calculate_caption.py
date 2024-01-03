
from pycocoevalcap.bleu.bleu import Bleu
import nltk
# if 'wordnet' not in nltk.corpus.reader.wordnet.__dict__:    
#     nltk.download('wordnet')
from nltk.translate import meteor_score
from rouge import Rouge
from tqdm import tqdm
import os
import json
import logging

def calculate_cap(json_name):
    json_file = '/data1/yubo/risk/GRiT/eval/'
    log_path = '/data1/yubo/risk/GRiT/eval/calculate_caption/'

    log_name = json_name
    log_name = log_name.split('.')[0] + '.log'
    # 配置日志模块 
    log_path = open(os.path.join(log_path, log_name), encoding="utf-8", mode="w")
    logging.basicConfig(
        stream=log_path,
        datefmt = '%d-%m-%Y %H%M%S',
        format = '%(asctime)s %(name)s: %(levelname)s: %(message)s',
        level = logging.INFO
    )
    logging.info('\n开始执行{}'.format(log_path))

    with open(os.path.join(json_file, 'predict_json', json_name), 'r') as f:
        datas = json.load(f)
    f.close()

    total = 0
    count_has1 = 0
    count_multi = 0
    no_caption = 0

    b1_score = []
    b4_score = []
    m_score = []
    r_score = []
    c_score = []
    s_score = []

    for data in tqdm(datas):
        total += 1
        pred_caption_len = len(data["pred_captions"])
        gt_caption = [data["gt_captions"]]
        if(pred_caption_len==0):
            no_caption += 1
            continue
        elif(pred_caption_len>1):
            count_multi += 1
            pred_caption = [data["pred_captions"][0]]
            continue
        elif(pred_caption_len==1):
            count_has1 += 1
            pred_caption = data["pred_captions"]

        # 加载参考文本
        references = {
            # "image1": ["there is a cat on the mat"],
            "image1": gt_caption,
        }
        # 加载生成文本
        hypotheses = {
            # "image1": ["the cat is on the mat"],
            "image1": pred_caption,
        }
    
        # 计算 BLEU 分数
        bleu_score = Bleu(n=4)
        scores, _ = bleu_score.compute_score(references, hypotheses)
        b1_score.append(scores[0])
        b4_score.append(scores[3])
        
        # 计算 METEOR 指标
        meteor = meteor_score.meteor_score([references["image1"][0].split()], hypotheses["image1"][0].split())
        m_score.append(meteor)
        
        # 计算 ROUGE 指标
        rouge = Rouge()
        rouge_scores = rouge.get_scores(hypotheses["image1"][0], references["image1"][0])
        r_score.append(rouge_scores[0]["rouge-1"]["p"])

        #计算 CIDER 指标
        import sys
        sys.path.insert(0,'/data1/yubo/risk/CaptionMetrics/')
        from pycocoevalcap.cider.cider import Cider    
        cider = Cider()
        (cider_score, cider_scores) = cider.compute_score(hypotheses, references)
        c_score.append(cider_score)

    # 打印 BLEU 分数
    logging.info(f"no_caption={no_caption}")
    logging.info(f"data[pred_object_descriptions]=1,{count_has1}")
    logging.info(f"data[pred_object_descriptions]>1,{count_multi}")
    logging.info(f"total={total}")

    logging.info(f"BLEU-1={sum(b1_score)/count_has1}")
    logging.info(f"BLEU-4={sum(b4_score)/count_has1}")
    logging.info(f"METEOR={sum(m_score)/count_has1}")
    logging.info(f"ROGUE={sum(r_score)/count_has1}")
    logging.info(f"CIDER={sum(c_score)/total}")
    # logging.info(f"SPICE={sum(s_score)/total}")
    logging.info('结束{}\n'.format(log_path))
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

    calculate_cap(json_name)

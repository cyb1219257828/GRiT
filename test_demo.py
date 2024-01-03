# import sys
# sys.path.insert(1,'/data1/yubo/risk/CaptionMetrics/')
# from pycocoevalcap.cider.cider import Cider    

# scorer = Cider()

# gts = {"184321": ["train traveling down a track in front of"]}
# res = {"184321": ["train traveling down a track in front of a road"]}

# (score, scores) = scorer.compute_score(gts, res)
# print('cider = %s' % score)


from pycocoevalcap.cider.cider import Cider    
scorer = Cider()
gts = {"184321": ["a train traveling down tracks next to lights"]}
res = {"184321": ["train traveling down a track in front of a road"]}
(score, scores) = scorer.compute_score(gts, res)
print('cider = %s' % score)

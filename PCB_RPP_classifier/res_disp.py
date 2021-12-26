# 使用肉眼测试分类器性能, 展示
# Programed by Z.Zhao

import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from mylib4test import *

res_gt = './result/features_tiny_full_gt.npy'
# fail in q_8_c3
res_gt_ft = './result/features_tiny_full_gt_ft.npy'
# fail in q_8_c3,


res_det = './result/features_tiny_full_det.npy'
# fail in q_1_c6,  q_8_c3, q_8_c6
res_det_ft = './result/features_tiny_full_det_ft.npy'
# fail in q_8_c3


res_path = res_det
is_show_all = True

if __name__ == "__main__":
    # test_opt = TestConfig()
    # test_opt.print_info()
    # dataloaders, img_datasets, ql, qc, gl, gc = load_test_data(test_opt)
    result = np.load(res_path, allow_pickle=True).item()
    qf, gf = torch.FloatTensor(result['qf']), torch.FloatTensor(result['gf'])
    qf = qf.cuda()
    gf = gf.cuda()

    img_datasets = result['img_datasets']
    gl, gc = get_lc(img_datasets['gallery'].imgs)
    ql, qc = get_lc(img_datasets['query'].imgs)

    num_q = len(ql)
    if is_show_all:
        for i in range(5, 6):
            path_q, _ = img_datasets['query'].imgs[i]
            hit_img(path_q, img_datasets, gf=gf, gl=gl, gc=gc, qf=qf[i], ql=ql[i], qc=qc[i])
    else:
        ids_q = random.sample(range(num_q), 5)
        for i in ids_q:
            path_q, _ = img_datasets['query'].imgs[i]
            hit_img(path_q, img_datasets, gf=gf, gl=gl, gc=gc, qf=qf[i], ql=ql[i], qc=qc[i])














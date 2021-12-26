# 根据features, 计算CMC & mAP
# Programed by Z.Zhao

from mylib4test import *
import torch
import numpy as np

res_gt = './result/features_tiny_full_gt.npy'
# fail in q_8_c3
res_gt_ft = './result/features_tiny_full_gt_ft.npy'
# fail in q_8_c3,


res_det = './result/features_tiny_full_det.npy'
# fail in q_1_c6,  q_8_c3, q_8_c6
res_det_ft = './result/features_tiny_full_det_ft.npy'
# fail in q_8_c3


res_std_gt = './result/features_std_full_gt.npy'


if __name__ == "__main__":
    torch.cuda.set_device(0)
    # print('gt  no_ft')
    # eval_test(res_gt)
    # print('\ngt  ft')
    # eval_test(res_gt_ft)
    # print('\ndet no_ft')
    # eval_test(res_det)
    # print('\ndet ft')
    # eval_test(res_det_ft)
    eval_test(res_std_gt)









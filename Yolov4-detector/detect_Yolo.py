# 读取数据集的图片, 使用Yolov4检测该图片的所有的物品, 标注出所有行人并生成txt文件
# Programed by Z.Zhao

import os
import shutil
import cv2
from tqdm import tqdm, trange
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet


use_cuda = True
img_path = 'D:/Research/ReID/ReID_Yolo_PCB/Dataset/UESTC_ReID/img/'
res_path = './res_detect/'

if __name__ == '__main__':

    if not os.path.exists(res_path):
        os.mkdir(res_path)
    else:
        shutil.rmtree(res_path)
        os.mkdir(res_path)

    cfg_path = './data/yolov4.cfg'
    weight_path = './data/yolov4.weights'
    m = Darknet(cfg_path)
    m.load_weights(weight_path)
    if use_cuda:
        m.cuda()
    print('Load weights, done!')

    cams = ["cam" + '%d' % i for i in range(1, 7)]
    frames = [[0, 599], [25, 591], [0, 600], [0, 599], [0, 597], [0, 599]]  # 有人的画面

    for idx_cam in range(6):
        with open(res_path + cams[idx_cam] + '.txt', 'a') as res_file:  # 文件不存在, a模式自动创建
            for i in trange(frames[idx_cam][0], frames[idx_cam][0]+5):
                img = cv2.imread(img_path + cams[idx_cam] + '/%d.jpg' % i)
                sized_img = cv2.resize(img, (m.width, m.height))
                boxes = do_detect(m, sized_img, 0.4, 0.6, use_cuda)  # resize & detect
                # x1=box[0]*width  y1=box[1]*height  x2=box[2]*width  y2=box[3]*height cls_conf = box[5] cls_id = box[6]
                for box in boxes[0]:
                    if box[6]:  # 不是行人直接跳过
                        break
                    cen_x = int((box[0] + box[2]) / 2 * img.shape[1])
                    cen_y = int((box[1] + box[3]) / 2 * img.shape[0])
                    w = int((box[2] - box[0]) * img.shape[1])
                    h = int((box[3] - box[1]) * img.shape[0])
                    # 以frame, cx, xy, w, h标准储存
                    res_file.write(str(i) + ',' + str(cen_x) + ',' + str(cen_y) + ',' + str(w) + ',' + str(h) + '\n')













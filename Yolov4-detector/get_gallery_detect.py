# 根据 res_detect给出的label, 在img中抠出图像, 命名规则: /id/id_camx_xxx.jpg  xxx为frame, id为两位数
# Programed by Z.Zhao

import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    img_path = 'D:/Research/ReID/ReID_Yolo_PCB/Dataset/UESTC_ReID/img/'
    label_path = './res_label/'
    save_path = "D:/Research/ReID/ReID_Yolo_PCB/Dataset/UESTC_ReID/UESTC_ReID_det_std/gallery/"

    cams = ["cam" + '%d' % i for i in range(1, 7)]
    ids = ["%02d" % i for i in range(1, 18)]

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    for id in ids:
        save_path_id = os.path.join(save_path, id)
        os.mkdir(save_path_id)

    for cam in cams:
        img_path_cam = os.path.join(img_path, cam)
        label_path_cam = os.path.join(label_path, cam + ".txt")
        with open(label_path_cam) as label_cam:
            label_data = label_cam.readlines()
            # frame_idx, id, cen_x, cen_y, w, h
            # 根据label_data截取图像并按照id存储
            for line_label in tqdm(label_data):
                line_label = line_label.strip('\n')  # 去除读取出的'\n'
                frame_idx, id, cen_x, cen_y, w, h = list(map(int, line_label.split(',')))  # 分割并转化为数字列表
                frame_path = os.path.join(img_path_cam, '%d' % frame_idx + '.jpg')  # 读取frame图像
                img_frame = cv2.imread(frame_path)  # 返回RGB三维数组
                img_frame_id = img_frame[cen_y - h//2 : cen_y + h//2, cen_x - w//2: cen_x + w//2]
                if img_frame_id.size != 0:
                    save_path_img = os.path.join(save_path, "%02d/%02d_%s_%d.jpg" % (id, id, cam, frame_idx))
                    cv2.imwrite(save_path_img, img_frame_id)
                # else:
                #     print(cam + '-error: ' + line_label)







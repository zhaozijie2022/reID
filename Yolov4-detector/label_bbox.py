# 对 get_bbox 得到的 bbox 打标签: 计算与 gt 的 iou 并给出标签
# Programed by Z.Zhao

import os
import shutil
import numpy as np
from tqdm import tqdm, trange


gt_path = 'D:/Research/ReID/ReID_Yolo_PCB/Dataset/UESTC_ReID/label_check/'
bbox_path = './res_detect/'
iou_bias = 0.6
res_path = './res_label/'


def compute_iou(bbox_gt, bbox_res):
    bbox_gt_xy = [bbox_gt[0] - bbox_gt[2]/2, bbox_gt[1] - bbox_gt[3]/2, bbox_gt[0] + bbox_gt[2]/2, bbox_gt[1] + bbox_gt[3]/2]
    bbox_res_xy = [bbox_res[0] - bbox_res[2]/2, bbox_res[1] - bbox_res[3]/2, bbox_res[0] + bbox_res[2]/2, bbox_res[1] + bbox_res[3]/2]

    max_x = max(bbox_gt_xy[2], bbox_res_xy[2])
    min_x = min(bbox_gt_xy[0], bbox_res_xy[0])
    width = bbox_gt[2] + bbox_res[2] - (max_x - min_x)
    max_y = max(bbox_gt_xy[3], bbox_res_xy[3])
    min_y = min(bbox_gt_xy[1], bbox_res_xy[1])
    height = bbox_gt[3] + bbox_res[3] - (max_y - min_y)
    intersection = width * height
    union = bbox_gt[2] * bbox_gt[3] + bbox_res[2] * bbox_res[3] - intersection

    return intersection / union


if __name__ == '__main__':
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    else:
        shutil.rmtree(res_path)
        os.mkdir(res_path)

    cams = ["cam" + '%d' % i for i in range(1, 7)]
    for idx_cam in range(6):
        # 将txt文件的detect_bbox和gt_bbox转化为array
        with open(gt_path + cams[idx_cam] + '.txt', 'r') as gt_file, open(bbox_path + cams[idx_cam] + '.txt', 'r') as detect_file:
            gt_lines = gt_file.readlines()
            detect_lines = detect_file.readlines()
            # 将cam.txt读取为array
            gt_data = []
            for gt_line in gt_lines:
                gt_line = gt_line.strip('\n')  # 删除本行换行符
                frame_idx, target_id, cen_x, cen_y, w, h = list(map(int, gt_line.split(',')))
                gt_data.append([frame_idx, target_id, cen_x, cen_y, w, h])
            gt_data = np.array(gt_data)

            detect_data = []
            for detect_line in detect_lines:
                detect_line = detect_line.strip('\n')
                frame_idx, cen_x, cen_y, w, h = list(map(int, detect_line.split(',')))
                detect_data.append([frame_idx, cen_x, cen_y, w, h])
            detect_data = np.array(detect_data)

        # 根据bbox和gt的iou给detect打标签
        with open(res_path + cams[idx_cam] + '.txt', 'a') as res_file:
            res = []
            for i in range(detect_data.shape[0]):  # 对bbox遍历
                bbox = detect_data[i, :]
                find_range = np.where(gt_data[:, 0] == bbox[0])[0]  # 寻找与当前bbox同帧的gt_box
                gt_test = gt_data[find_range, :]  # 同一帧的所有gt_bbox
                if len(gt_test) == 0:
                    continue
                dist = (gt_test[:, 2] - bbox[1]) ** 2 + (gt_test[:, 3] - bbox[2]) ** 2  # 计算当前bbox与所有待检测gt_bbox中心点的距离
                pos = np.argmin(dist)
                target = gt_test[pos, :]  # 与bbox距离最近的gt_bbox
                iou = compute_iou(bbox[1:], target[2:])  # 计算iou
                if iou >= iou_bias:  # 认为配对成功
                    id_bbox = target[1]
                    res_file.write(str(bbox[0]) + ',' + str(id_bbox) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ',' + str(bbox[4]) + '\n')
                    res.append([bbox[0], id_bbox, bbox[1], bbox[2], bbox[3], bbox[4]])


































# 从detect和gt两种gallery中随机采样相同名称的img展示, 查看detect效果
# Programed by Z.Zhao

import os
import random
import cv2
import matplotlib.pyplot as plt

print(os.path.abspath(__file__))
gt_path = 'D:/Research/ReID/ReID_Yolo_PCB/Dataset/UESTC_ReID/UESTC_ReID_gt_std/gallery/'
detect_path = 'D:/Research/ReID/ReID_Yolo_PCB/Dataset/UESTC_ReID/UESTC_ReID_det_std/gallery/'
num_img = 6

if __name__ == "__main__":
    ids = ["%02d" % i for i in range(1, 18)]
    img_list = []
    for id in ids:
        img_list += os.listdir(detect_path + id)
    img_idxs = random.sample(img_list, num_img)  # 读取图片并采样 xx_camx_*.jpg

    fig = plt.figure('test detection')
    for i in range(num_img):
        img_path = os.path.join(gt_path, img_idxs[i][0:2] + '/' + img_idxs[i])
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        ax = fig.add_subplot(3, 4, i*2+1)
        ax.imshow(img)
        plt.axis('off')
        title = img_idxs[i].strip('.jpg')
        plt.title('gt_%s_c%s' % (title[0:2], title[6:]))

        img_path = os.path.join(detect_path, img_idxs[i][0:2] + '/' + img_idxs[i])
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        ax = fig.add_subplot(3, 4, i*2+2)
        ax.imshow(img)
        plt.axis('off')
        title = img_idxs[i].strip('.jpg')
        plt.title('det_%s_c%s' % (title[0:2], title[6:]))
    plt.show()
    print(img_idxs)







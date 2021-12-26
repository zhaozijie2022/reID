# reid_course_project_term7

一、下载数据集
1. Market-1501-v15.09.15
2. UESTC-ReID
下载后将他们放在./Dataset/下
在UESTC-ReID中：

check_label.py用于提出gt中的坏bbox，并将结果储存在label_check文件夹中；

query_reorganize.py用于将query重新排列为torch.utils.data.Dataloader规定的形式：./id/imgs.jpg

get_gallery_gt.py根据label从逐帧的img中抠出行人图像，命名规则为：gallery/id/id_camx_xxx.jpg id为两位数（01-17），xxx为第几帧

sample_gallery.py从每个id每个cam中采样num_img张图片，减小gallery规模以加快test速度

其余文件为老师下发的数据集中自带的文件

二、下载权重

Yolov4：https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

下载后放在/Yolov4-detector/data/下

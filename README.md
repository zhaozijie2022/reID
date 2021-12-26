# reid_course_project_term7

一、下载数据集及权重
1. Market-1501-v15.09.15.zip：https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ   to   ./Dataset
2. yolov4.weights：https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT  to  ./Yolov4-detector/data/
3. UESTC-ReID：https://pan.baidu.com/s/1VZPfZOT2Ig6-rZD04Fx5zw 提取码：reid  to  ./Dataset/UESTC-ReID

二、文件介绍
1. Yolov4-detector


在UESTC-ReID中：

  check_label.py用于提出gt中的坏bbox，并将结果储存在label_check文件夹中；

  query_reorganize.py用于将query重新排列为torch.utils.data.Dataloader规定的形式：./id/imgs.jpg

  get_gallery_gt.py根据label从逐帧的img中抠出行人图像，命名规则为：gallery/id/id_camx_xxx.jpg id为两位数（01-17），xxx为第几帧

  sample_gallery.py从每个id每个cam中采样num_img张图片，减小gallery规模以加快test速度

  压缩包中不包含老师下发的文件



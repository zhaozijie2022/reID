# reid_course_project_term7
使用Yolov4+PCB_RPP算法完成行人重识别课程设计

# 下载数据集及权重
1. Market-1501-v15.09.15.zip：https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ   to   ./Dataset
2. yolov4.weights：https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT  to  ./Yolov4-detector/data/
3. UESTC-ReID：https://pan.baidu.com/s/1VZPfZOT2Ig6-rZD04Fx5zw （提取码：reid）  to  ./Dataset/UESTC-ReID

# 文件介绍
1. Dataset/UESTC-ReID：

    check_label.py用于提出gt中的坏bbox，并将结果储存在label_check文件夹中 \
    query_reorganize.py用于将query重新排列为torch.utils.data.Dataloader规定的形式：./id/imgs.jpg \
    get_gallery_gt.py根据label从逐帧的img中抠出行人图像，命名规则为：gallery/id/id_camx_xxx.jpg id为两位数（01-17），xxx为第几帧 \
    sample_gallery.py从每个id每个cam中采样num_img张图片，减小gallery规模以加快test速度
 
2. Yolov4-detector：
    
    detect_Yolo.py  读取数据集的图片, 使用Yolo检测，标注出所有行人并生成txt文件（img_path, res_path） \ 
    label_bbox.py  对 detect_Yolo 得到的 bbox 打标签: 计算与 gt 的 iou 并给出标签 （gt_path, bbox_path, iou_bias, res_path） \
    get_gallery_detect.py  根据res_detect给出的label, 在img中抠出图像, 命名规则同上（img_path, label_path, save_path） \
    res_disp.py  从detect和gt两种gallery中随机采样相同名称的img展示, 查看detector效果 \
    models.py from https://github.com/Tianxiaomo/pytorch-YOLOv4

3. PCB_RPP-classifier
    
    mylib4test.py  test及fine tune所需的函数库，具体见注释 \
    get_features.py 对gallery和query提取特征, 并使用npy格式存储 \
    res_disp.py  随机抽取num_img张query，展示效果 \
    evaluate_test.py 根据features, 计算CMC & mAP
    train.py, prepare.py & model.py from https://github.com/Xiaoccer/ReID-PCB_RPP
 
# 效果展示



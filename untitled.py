from ultralytics import YOLO, RTDETR, YOLOWorld
import os

# ---------------------存在问题---------------------------
# 1.正负样本分布及其不均衡
# 2.各类标签之间的分布及其不均衡
# 3.大部分特征为小目标特征

# -------------------------7月1 测试------------------
# train2 是yolov8s 100epoch  non-pretrained
# train3 是yolov8m 100epoch  non-pretrained mAP0.5=0.208左右 得分为 0.35
# train4 是接着train3的best.pt 继续训练100peoch 还是yolov8m  ---感觉效果越来越差了 不用跑了
# train5 试一下yolov10m， 使用与训练权重，看看效果  0.36

# -------------7.2测试-----------------
# train6 yolov8s.pt  看起来效果挺好的，还没有去测试  应该是早停了
# train7 yolov8s.pt 减少数据集中的background，接近1：1 200 epoch 效果不是很好  废弃
# train8 yolov8s.pt 更换验证集，效果还挺好的  还没有输出output文件，这个效果最好0.4

# -------------7.3测试--------------------------------
# train11 yolov5s.pt 看起来效果会很好
# train9 rt-detr 效果还可以0.4056
# trian8 conf=0.2效果降低
# train8 conf=0.4 效果降低
# train12 yolov8s.pt 修改数据集，加上所有的背景，看一下效果如何 还没测试 0.37效果不是很好
# train13 yolov8s 将缩放因子scale =1，就是不对图片进行缩放，看看正确率如何。

# -------------------7.23结果分析--------------------------
# 从结果上来看，V的效果好，因为检测出5的少，因此效果比较好
# 2和7的效果不好，是因为类别5的数量大，导致效果不好
# 可以尝试一下不检测类别5，看一下效果如何

def txt_to_json(log='train',Pre=False):

    # 首先判断一些文件夹是否存在
    folders = ['output2', 'output7', 'outputV']
    for fo in folders:
        folder_path = os.path.join('runs/detect', fo)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            os.system(f"rm -r {folder_path}")
        else:
            print(f"The folder '{folder_path}' does not exist.")


    # ----------------对结果进行预测----------------------
    output2 = 'output2'
    output7 = 'output7'
    outputV = 'outputV'

    result_path2 = r'/root/autodl-tmp/ultralytics/json_result/2.json'
    result_path7 = r'/root/autodl-tmp/ultralytics/json_result/7.json'
    result_pathV = r'/root/autodl-tmp/ultralytics/json_result/V.json'

    image_dir2 = f'/root/autodl-tmp/ultralytics/runs/detect/{output2}/'
    image_dir7 = f'/root/autodl-tmp/ultralytics/runs/detect/{output7}/'
    image_dirV = f'/root/autodl-tmp/ultralytics/runs/detect/{outputV}/'
    
    if Pre:
        test_2 = r"/root/autodl-tmp/PrepostTest/2/Images/"
        test_7 = r"/root/autodl-tmp/PrepostTest/7/Images/"
        test_V = r"/root/autodl-tmp/PrepostTest/V/Images/"
    else:
        test_2 = r"/root/autodl-tmp/Testset/2/Images/"
        test_7 = r"/root/autodl-tmp/Testset/7/Images/"
        test_V = r"/root/autodl-tmp/Testset/V/Images/"
    model = RTDETR(f"/root/autodl-tmp/ultralytics/runs/detect/{log}/weights/best.pt")
    # model = YOLO(f"/root/autodl-tmp/ultralytics/runs/detect/{log}/weights/best.pt")
    
    result = model.predict(source=test_2, save=True, save_conf=True, classes=[0,1,2,3,4,5], save_txt=True, name=f'{output2}')
    result = model.predict(source=test_7, save=True, save_conf=True, save_txt=True, name=f'{output7}')
    result = model.predict(source=test_V, save=True, save_conf=True, save_txt=True, name=f'{outputV}')

    os.system(f"python convert.py --save_json {result_path2} --image_dir {image_dir2} --predtxt_dir {image_dir2}labels")
    os.system(f"python convert.py --save_json {result_path7} --image_dir {image_dir7} --predtxt_dir {image_dir7}labels")
    os.system(f"python convert.py --save_json {result_pathV} --image_dir {image_dirV} --predtxt_dir {image_dirV}labels")
    print("-------------结果保存完毕----------------")
  

if __name__ == "__main__":

    # ----------------训练模型 --------------------
    # model = YOLO("yolov8l.pt",)
    # model.train(data="yga.yaml", epochs=200, batch=8, scale=1)  # 训练模型

    
    # --------------RTDETR模型训练---------------------------
    # from ultralytics import YOLO, RTDETR
    # model = RTDETR('rtdetr-l.pt')
    # model.train(data="yga.yaml", epochs=200, batch=24)
    
    # model = YOLOWorld('yolov8s-worldv2.pt')
    # # # model = YOLOWorld("yolov8m-worldv2.pt")
    # model.train(data="yga.yaml", epochs=200, batch=16)
    
    # model = YOLO('yolov8s.pt')  
    # model.train(data="yga2.yaml", epochs=200, batch=24)
    
    # -----------------------------7.23-----------------------------
    # yolov8s 使用新的数据集，效果比之前的好
    # model = YOLO('/root/autodl-tmp/ultralytics/runs/detect/train25/weights/best.pt')
    # model.train(data="yga2.yaml", epochs=200, batch=24, resume=True)
    # model.train(data="yga2.yaml", epochs=200, batch=48)
    
    # -------------------------7.23-----------------------------
    # 对训练后的模型，只针对部分类别的数据进行训练,并增加数据增强
    # 在yolov8s最好的基础上，进行一些数据变化，然后进行训练
    # 只训练部分的效果可能不太好
    # model = YOLO('/root/autodl-tmp/ultralytics/runs/detect/train15/weights/best.pt')
    # model.train(data="yga2.yaml", epochs=100, batch=24, flipud=0.6, mixup=0.1, classes=[2,6,7,8,9],resume=True)

    # -------------------------7.23------------------------------
    # 在train25最好的基础上，剔除背景，只训练带标签的
    # 把希望寄托在这个上试一下 即使mAP很高，效果也不一定好
    # model = YOLO('/root/autodl-tmp/ultralytics/runs/detect/train25/weights/best.pt')
    # model.train(data="yga3.yaml", epochs=200, batch=24) 
                #  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                #    all        351        525      0.913      0.914      0.947      0.734
                #      0        162        177      0.966      0.952      0.986      0.718
                #      1        102        108      0.927      0.945      0.985      0.725
                #      2         26         26      0.961      0.955      0.981      0.788
                #      3         50         61       0.95      0.943      0.964      0.815
                #      4         39         39      0.967      0.974      0.979      0.832
                #      5         63         63       0.94      0.994      0.993      0.951
                #      6          9         17       0.85      0.706       0.77      0.367
                #      7          8          9      0.909      0.667      0.941      0.681
                #      8         19         19      0.974          1      0.995      0.895
                #      9          5          6      0.688          1      0.879      0.568
    

    # model = YOLO('yolov8n.pt')
    # model.train(data="yga2.yaml", epochs=200, batch=64)
    
    # model = YOLO('yolov8m.pt')
    # model.train(data="yga2.yaml", epochs=200, batch=24)
    # ---------------------7.24-------------------------------
    # 使用yolov8s继续训练，看一下收敛以后的效果
    # model = YOLO('yolov8s.pt')
    # model.train(data="yga2.yaml", epochs=500, batch=32)
    
    # -------------------7.26---------------------
    # 使用Rtdetr X的效果不错0.475 RTDERTR-L的效果也不错 train21效果最好 分别把2：1 7：1 V：0.98
    # 之后尝试一下把RTDETR训练完，看一下效果如何
    # 都是使用预训练权重进行训练的 train20
    
    # -------------------7.27---------------------
    # 没有载入与训练权重，试一下看看效果怎么样
    # 这个训练到88轮暂停了，训练时间较长，暂时中止，后续继续训练
    # model = RTDETR('rtdetr-resnet50.yaml')
    # model.train(data="yga2.yaml", epochs=200, batch=16)
    
    
    # 继续训练RTDETR-L模型,有使用预训练权重
    # model = RTDETR('/root/autodl-tmp/ultralytics/runs/detect/train20/weights/last.pt')
    # model.train(data="yga2.yaml", epochs=200, batch=16, resume=True)
    
    # 实验了一下 v8m-cls：2:1 7:1 V：0.96 2heV提升了，7下降了
    # yolov8s-cls效果更好，RTDETR效果也可以很好
    
    
    # -------------7.28-----------------
    # model = RTDETR('/root/autodl-tmp/ultralytics/runs/detect/train20/weights/best.pt')
    # model.train(data="yga2.yaml", epochs=100, batch=16, mixup=0.4, flipud=0.5, bgr=0.4, copy_paste=0.4)
    
    
    
    # TO DO
    # 1.扩增数据集
    # 2。


    # --------将结果转化为 从txt格式转换为json格式----------
    txt_to_json(log='train20', Pre=True)
    
    
  


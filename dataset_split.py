import os
import random
import numpy as np
# 设置随机种子
random.seed(1225)
np.random.seed(1225)
"""
划分数据集 训练集和验证集的比例为0.1:0.9



"""

# os.system("mkdir -p ./datasets/train/images")
# os.system("mkdir -p ./datasets/train/labels")
# os.system("mkdir -p ./datasets/val/images")
# os.system('mkdir -p ./datasets/val/labels')

# --------------------创建分类数据集-----------------
os.system("mkdir -p ./datasets/train/NG")
os.system("mkdir -p ./datasets/train/OK")
os.system("mkdir -p ./datasets/val/NG")
os.system("mkdir -p ./datasets/val/OK")

# def move(img):
#     for im in img:
#         label_number = im.split(".")[0]
#         tmp_l = f'{label_number}.txt'
#         im_path = os.path.join(image_path, im)
#         lb_path = ps.path.join(label_path, tmp_l)


choose = ['NG', 'OK']

color_file = os.listdir('Trainset22')
# print(color_file)
for color in color_file:
    color_path = os.path.join("Trainset22", color)
    print(color_path)
    size_file = os.listdir(color_path)
    print(size_file)
    for s in size_file:
        size_path = os.path.join(color_path, s)
        print(size_path)
        for c in choose:
# ----------------------------------目标检测数据集--------------------------------------------------
#             if c == 'NG':
#                 # pass
#                 # 需要对这些图片文件进行重命名，方法：可以加上型号和好快作为区别的标签
#                 ppath = os.path.join(size_path, c)
#                 image_path = os.path.join(ppath, 'images')
#                 label_path = os.path.join(ppath, 'labels')
#                 # print(image_path)
#                 img = os.listdir(image_path)
#                 img_len = len(img)
#                 print(img_len)
#                 train_img = np.random.choice(img, int(0.9*img_len), replace=False)
#                 # train_img = np.random.choices(im)
#                 # print(train_img)
#                 # print(len(train_img))
#                 val_img = [x for x in img if x not in train_img]
#                 print(len(val_img))
#                 for im in train_img:
#                     label_number = im.split(".")[0]
#                     tmp_l = f'{label_number}.txt'
#                     im_path = os.path.join(image_path, im)
#                     lb_path = os.path.join(label_path, tmp_l)
#                     os.system(f'cp {im_path} ./datasets/train/images/{s}_{c}_{label_number}.jpg')
#                     os.system(f'cp {lb_path} ./datasets/train/labels/{s}_{c}_{label_number}.txt')

#                 for im in val_img:
#                     label_number = im.split(".")[0]
#                     tmp_l = f'{label_number}.txt'
#                     im_path = os.path.join(image_path, im)
#                     lb_path = os.path.join(label_path, tmp_l)
#                     os.system(f'cp {im_path} ./datasets/val/images/{s}_{c}_{label_number}.jpg')
#                     os.system(f'cp {lb_path} ./datasets/val/labels/{s}_{c}_{label_number}.txt')

                # print(img)

#             if c == 'OK':
#                 # OK 没有imgas和labels之分，没有背景的数据集生成一份
#                 image_path = os.path.join(size_path, c)
#                 # image_path = os.path.join(ppath, 'images')
#                 img = os.listdir(image_path)
#                 if '.ipynb_checkpoints' in img:
#                     img.remove('.ipynb_checkpoints')
#                 img_len = len(img)
#                 train_img = np.random.choice(img, int(0.9*img_len),replace=False)
#                 val_img = [x for x in img if x not in  train_img]
#                 for im in train_img:
#                     label_number = im.split(".")[0]
#                     tmp_l = f'{label_number}.txt'
#                     im_path = os.path.join(image_path, im)
#                     lb_path = os.path.join(label_path, tmp_l)
#                     os.system(f'cp {im_path} ./datasets/train/images/{s}_{c}_{label_number}.jpg')
#                     # os.system(f'cp {im_path} ./datasets/train/labels/{s}_{c}_{label_number}.txt')

#                 for im in val_img:
#                     label_number = im.split(".")[0]
#                     tmp_l = f'{label_number}.txt'
#                     im_path = os.path.join(image_path, im)
#                     lb_path = os.path.join(label_path, tmp_l)
#                     os.system(f'cp {im_path} ./datasets/val/images/{s}_{c}_{label_number}.jpg')
#                     # os.system(f'cp {im_path} ./datasets/val/labels/{s}_{c}_{label_number}.txt')



#   -----------------------------------------分类数据集-------------------------------------------------

            if c == 'NG':
                # pass
                # 需要对这些图片文件进行重命名，方法：可以加上型号和好快作为区别的标签
                ppath = os.path.join(size_path, c)
                image_path = os.path.join(ppath, 'images')
                label_path = os.path.join(ppath, 'labels')
                # print(image_path)
                img = os.listdir(image_path)
                img_len = len(img)
                print(img_len)
                train_img = np.random.choice(img, int(0.9*img_len), replace=False)
                # train_img = np.random.choices(im)
                # print(train_img)
                # print(len(train_img))
                val_img = [x for x in img if x not in train_img]
                print(len(val_img))
                for im in train_img:
                    label_number = im.split(".")[0]
                    tmp_l = f'{label_number}.txt'
                    im_path = os.path.join(image_path, im)
                    lb_path = os.path.join(label_path, tmp_l)
                    os.system(f'cp {im_path} ./datasets/train/NG/{s}_{c}_{label_number}.jpg')
                    # os.system(f'cp {lb_path} ./datasets/train/labels/{s}_{c}_{label_number}.txt')

                for im in val_img:
                    label_number = im.split(".")[0]
                    tmp_l = f'{label_number}.txt'
                    im_path = os.path.join(image_path, im)
                    lb_path = os.path.join(label_path, tmp_l)
                    os.system(f'cp {im_path} ./datasets/val/NG/{s}_{c}_{label_number}.jpg')
                    # os.system(f'cp {lb_path} ./datasets/val/labels/{s}_{c}_{label_number}.txt')

                # print(img)

            if c == 'OK':
                # OK 没有imgas和labels之分，没有背景的数据集生成一份
                image_path = os.path.join(size_path, c)
                # image_path = os.path.join(ppath, 'images')
                img = os.listdir(image_path)
                if '.ipynb_checkpoints' in img:
                    img.remove('.ipynb_checkpoints')
                img_len = len(img)
                train_img = np.random.choice(img, int(0.9*img_len),replace=False)
                val_img = [x for x in img if x not in  train_img]
                for im in train_img:
                    label_number = im.split(".")[0]
                    tmp_l = f'{label_number}.txt'
                    im_path = os.path.join(image_path, im)
                    lb_path = os.path.join(label_path, tmp_l)
                    os.system(f'cp {im_path} ./datasets/train/OK/{s}_{c}_{label_number}.jpg')
                    # os.system(f'cp {im_path} ./datasets/train/labels/{s}_{c}_{label_number}.txt')

                for im in val_img:
                    label_number = im.split(".")[0]
                    tmp_l = f'{label_number}.txt'
                    im_path = os.path.join(image_path, im)
                    lb_path = os.path.join(label_path, tmp_l)
                    os.system(f'cp {im_path} ./datasets/val/OK/{s}_{c}_{label_number}.jpg')
                    # os.system(f'cp {im_path} ./datasets/val/labels/{s}_{c}_{label_number}.txt')
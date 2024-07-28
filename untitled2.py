# 进行数据分类，对好和不好的数据集先进行分类，然后在进行预测
from ultralytics import YOLO
import os




def judge_folder(folder_path):       
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 删除并创建
        print(f"The folder '{folder_path}' has already existed!!! DELETE！.")
        os.system(f"rm -r {folder_path}")
        os.system(f"mkdir -p {folder_path}")
    else:
        print(f"The folder '{folder_path}' does not exist.")
        os.system(f"mkdir -p {folder_path}")
            
            


def preTest(results, size='2'):
    for result in results:
        print(result)
        for key, value in result.names.items():
            if value == 'NG':
                if key == 1:
                    print("hello")
                    ppath = result.path.split("/")[-1]
                    # 如果检测到100%有问题的话
                    os.system(f'cp {result.path} ../PrepostTest/{size}/Images/{ppath}')
                # else:
                #     pass
                
                # break


                
def result(log='train'):
    
    test_2 = r"/root/autodl-tmp/Testset/2/Images/"
    test_7 = r"/root/autodl-tmp/Testset/7/Images/"
    test_V = r"/root/autodl-tmp/Testset/V/Images/"
    
    
    test = r'/root/autodl-tmp/Testset/2/Images/P002_1440.jpg'
    
    # ---------处理结果----------------------
    
    # ---------创建三个保存结果的文件夹--------------
    pre2 = '../PrepostTest/2/Images'
    pre7 = '../PrepostTest/7/Images'
    preV = '../PrepostTest/V/Images'
    
    judge_folder(pre2)
    judge_folder(pre7)
    judge_folder(preV)

    
    model = YOLO(f'/root/autodl-tmp/ultralytics/runs/classify/{log}/weights/best.pt')
    # 结果会保存到output2中
    results2 = model.predict(test_2, save=True, imgsz=640, name='output2')
    results7 = model.predict(test_7, save=True, imgsz=640, name='output7')
    resultsV = model.predict(test_V, save=True, imgsz=640, name='outputV')
    
    for result in results2:
        for j in result.probs.top5:
            if result.names[j] == "NG" and result.probs.data[j] >= 1:
                ppath = result.path.split("/")[-1]
                os.system(f"cp {result.path} {pre2}/{ppath}")
                
    for result in results7:
        for j in result.probs.top5:
            if result.names[j] == "NG" and result.probs.data[j] >= 1:
                ppath = result.path.split("/")[-1]
                os.system(f"cp {result.path} {pre7}/{ppath}")
    
    for result in resultsV:
        for j in result.probs.top5:
            if result.names[j] == "NG" and result.probs.data[j] >= 0.98:
                ppath = result.path.split("/")[-1]
                os.system(f"cp {result.path} {preV}/{ppath}")
                
    
    
# ---------------------------------------7.26日--------------------------------------------
if __name__ == "__main__":
    
    # ----------------------使用yolov8s进行分类----------------------
    # model = YOLO('yolov8m-cls.pt')
    # result = model.train(data='/root/autodl-tmp/datasets/', epochs=200, batch=24, imgsz=640)
    
    
    
    
    # ------------模型预测---------------
    
    # model = YOLO(f'path/to/best_weight')
    # model('data/to/need/predict')
    
    # 结果预测
    result()
                
                
                

        








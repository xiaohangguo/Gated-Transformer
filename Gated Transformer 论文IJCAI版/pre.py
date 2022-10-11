import torch
print('当前使用的pytorch版本：', torch.__version__)
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset

import numpy as np
import pandas as pd
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 选择要跑的模型
save_model_path = '/media/hang/data/GTN/Gated Transformer 论文IJCAI版/saved_model/mts(5) 78.18 batch=250.pkl'
file_name = save_model_path.split('/')[-1].split(' ')[0]
path = f'/media/hang/data/GTN/Gated Transformer 论文IJCAI版/predict_arr(2).mat'  # 拼装数据集路径
BATCH_SIZE = int(save_model_path[save_model_path.find('=')+1:save_model_path.rfind('.')])
test_dataset = MyDataset(path, 'test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# 加载模型
net = torch.load(save_model_path, map_location=torch.device('cuda'))

def test(dataloader, flag='test_set'):

    with torch.no_grad():
        net.eval()
        list1 = []
        for x ,y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre,_,_,_,_,_,_ = net(x,"test")
            score = y_pre.data
            #prob = torch.exp(score[:,0])/(torch.exp(score[:,0])+torch.exp(score[:,1]))
            score = score.tolist()
            list1 = list1 +score
        array1 = np.array(list1)[...,0].reshape(6234,1)
        print("数组元素总数：",array1.size)      #打印数组尺寸，即数组元素总数
        print("数组形状：",array1.shape)
        df = pd.read_csv("test_A.csv", header=None)
        array = np.array(df)
        array3 = np.hstack((array, array1))
        print("合并后的数组形状：",array3.shape)
        pd.DataFrame(array3,columns=["file_name","score"]).to_csv("sup_result.csv",index=None)
        print("导出成功！")
test(test_dataloader)
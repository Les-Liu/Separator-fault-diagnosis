"""
    数据集归一化处理
    1. 读取文件路径下所有工况的数据集
    2. 分别进行归一化处理
    3. 依次保存进行储存文件夹
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def LoadData(file_path:str):
    """
    @Description :
        读取指定文件夹下的所有Excel文件，文件后缀xlsx
    @Returns :
        工况数据集集合
    """
    datasets = {}

    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".xlsx"):
                file_path = os.path.join(root, file)
                file = file.replace(".xlsx", "")
                dataset = pd.read_excel(file_path)
                datasets[file] = dataset
    
    return datasets, dataset.columns

def SplitDataset(datasets:dict):
    """
    @Description :
        制作特征 X 和 标签 Y
    @Returns :
        所有工况的特征集合 和 标签集合
    """
    X = {}
    Y = {}
    for item in datasets:
        dataset = datasets[item]
        # 提取每个工况的标签
        Y[item] = dataset["标签"]
        # 提取每个工况的特征
        X[item] = dataset.drop(["标签"],axis=1)
    
    return X,Y

def Normalization(datasets:dict):
    """
    @Description :
        对数据进行归一化处理
    @Returns :
        数据归一化之后的结果
    """
    normalization_datasets = {}

    for item in datasets:
        dataset = datasets[item]
        transform = MinMaxScaler()
        normalization_dataset = transform.fit_transform(dataset)
        normalization_datasets[item] = normalization_dataset
    
    return normalization_datasets

def MergeData(X:dict,Y:dict):
    """
    @Description :
        将归一化的特征 X 和 标签 Y 进行合并
    @Returns :
        所有工况数据集合并的结果
    """
    all_datasets = {}

    for item in X:
        x = X[item]
        y = np.array(Y[item]).reshape(x.shape[0],1)
        dataset = pd.DataFrame(data=np.concatenate((x,y),axis=1),columns=columns_name)
        all_datasets[item] = dataset
    
    return all_datasets
        
if __name__ == '__main__':
    # 1.读取分离器系统所有工况数据
    file_path = 'F:/博士资料/本地论文/自己写的论文/第三篇论文/论文/数据/Train Data/Add Noise 50snr'
    datasets, columns_name = LoadData(file_path = file_path)
    # 2.制作特征数据集
    X,Y = SplitDataset(datasets = datasets)
    # 3.数据归一化处理
    normalization_datasets = Normalization(datasets=X)
    # 4.数据合并
    all_datasets = MergeData(X=normalization_datasets,Y=Y)
    # 5.将数据集保存到指定文件路径下
    specified_path = "F:/博士资料/本地论文/自己写的论文/第三篇论文/论文/数据/Train Data/Normalisation 50snr"
    for item in all_datasets:
        dataset = all_datasets[item]
        dataset.to_excel(specified_path + "/" + item + ".xlsx",index = False)
    


    

    
    


